#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

COMMAND=""
DATASET_REPO="Dexmal/so101_pick_cube"
DATA_ROOT="$PROJECT_ROOT/data/so101_pick_cube"
DATASET_DOWNLOAD_DIR="$PROJECT_ROOT/data/.hf_downloads/so101_pick_cube"
MODEL_REPO="${PRETRAIN_MODEL_REPO:-Dexmal/DM05}"
MODEL_DIR="$PROJECT_ROOT/checkpoints/DM05"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
FORCE=0
TRAIN_ARGS=()

usage() {
    cat <<EOF
Usage:
  script/so101_runner.sh <all|dataset|model|train> [options] [-- train args]

Subcommand is required. Use "all" to run every step.

Commands:
  dataset    Download and organize the Hugging Face SO101 dataset.
  model      Download the pretrained model from Hugging Face.
  train      Launch playground/dm05_so101_lora.py.
  all        Run dataset -> model -> train.

Options:
  --dataset-repo REPO       HF dataset repo. Default: Dexmal/so101_pick_cube
  --data-root DIR           Final SO101 data root. Default: ./data/so101_pick_cube
  --dataset-download-dir DIR
                            Temporary HF dataset download dir.
                            Default: ./data/.hf_downloads/so101_pick_cube
  --model-repo REPO         HF model repo. Default: Dexmal/DM05.
                            Can also use PRETRAIN_MODEL_REPO.
  --model-dir DIR           Final pretrained model dir. Default: ./checkpoints/DM05.
  --nproc-per-node N        torchrun processes per node. Default: \$NPROC_PER_NODE or 1
  --force                   Overwrite existing organized data files.
  -h, --help                Show this help.

Examples:
  script/so101_runner.sh dataset --data-root /path/to/data/so101_pick_cube
  script/so101_runner.sh model --model-repo org/model --model-dir /path/to/checkpoint
  script/so101_runner.sh train -- --trainer-config.num-train-steps 100
  script/so101_runner.sh all --nproc-per-node 8
EOF
}

log() {
    printf '[so101_runner] %s\n' "$*"
}

die() {
    printf '[so101_runner][error] %s\n' "$*" >&2
    exit 1
}

abspath() {
    python - "$1" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

have_command() {
    command -v "$1" >/dev/null 2>&1
}

dir_has_entries() {
    [[ -d "$1" && -n "$(find "$1" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]
}

extracted_so101_complete() {
    local dir="$1"
    [[ -d "$dir/so101_pick_cube/jsonl" && -d "$dir/so101_pick_cube/videos" ]]
}

so101_dataset_complete() {
    local dir="$1"
    [[ -d "$dir/jsonl" && -d "$dir/videos" ]]
}

hf_download() {
    local repo_type="$1"
    local repo="$2"
    local target_dir="$3"

    mkdir -p "$target_dir"
    if have_command hf; then
        hf download "$repo" \
            --repo-type "$repo_type" \
            --local-dir "$target_dir"
    elif have_command huggingface-cli; then
        huggingface-cli download "$repo" \
            --repo-type "$repo_type" \
            --local-dir "$target_dir"
    else
        die "Neither hf nor huggingface-cli is available. Install huggingface_hub first."
    fi
}

extract_archives() {
    local src_dir="$1"
    local extracted_dir="$src_dir/.extracted"
    local found=0

    mkdir -p "$extracted_dir"
    while IFS= read -r first_part; do
        found=1
        local base
        base="${first_part%.part-000}"
        local name
        name="$(basename "$base")"
        local out="$extracted_dir/$name"
        if extracted_so101_complete "$out" && [[ "$FORCE" -ne 1 ]]; then
            log "Keeping existing complete extracted archive $out (use --force to overwrite)"
            continue
        fi
        if dir_has_entries "$out"; then
            log "Removing incomplete extracted archive $out"
        fi
        rm -rf "$out"
        mkdir -p "$out"
        log "Extracting split archive ${base}.part-* to $out"
        find "$(dirname "$first_part")" -maxdepth 1 -type f -name "$(basename "$base").part-*" \
            | sort \
            | xargs cat \
            | tar -xf - -C "$out"
    done < <(find "$src_dir" -maxdepth 1 -type f -name '*.tar.part-000')

    while IFS= read -r archive; do
        found=1
        local name
        name="$(basename "$archive")"
        local out="$extracted_dir/${name%.*}"
        mkdir -p "$out"
        case "$archive" in
            *.zip)
                have_command unzip || die "unzip is required to extract $archive"
                unzip -q -o "$archive" -d "$out"
                ;;
            *.tar|*.tar.gz|*.tgz|*.tar.bz2|*.tbz2|*.tar.xz|*.txz)
                tar -xf "$archive" -C "$out"
                ;;
        esac
    done < <(find "$src_dir" -type f \( \
        -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o \
        -name '*.tar.bz2' -o -name '*.tbz2' -o -name '*.tar.xz' -o -name '*.txz' \
    \))

    if [[ "$found" -eq 1 ]]; then
        log "Extracted dataset archives under $extracted_dir"
    fi
}

copy_tree() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"
    if [[ -e "$dst" && "$FORCE" -ne 1 ]]; then
        log "Keeping existing $dst (use --force to overwrite)"
        return 0
    fi
    if have_command rsync; then
        mkdir -p "$dst"
        rsync -a "$src"/ "$dst"/
    else
        rm -rf "$dst"
        cp -a "$src" "$dst"
    fi
}

organize_so101_dataset() {
    local src_dir="$1"
    local dst_root="$2"
    local candidates=()

    mkdir -p "$dst_root"
    candidates+=("$src_dir")
    [[ -d "$src_dir/data/so101_pick_cube" ]] && candidates+=("$src_dir/data/so101_pick_cube")
    [[ -d "$src_dir/.extracted" ]] && candidates+=("$src_dir/.extracted")

    local copied=0
    local candidate
    local match

    # Try to find so101_pick_cube directory
    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate/so101_pick_cube" ]]; then
            if so101_dataset_complete "$candidate/so101_pick_cube"; then
                copy_tree "$candidate/so101_pick_cube" "$dst_root"
                copied=1
                break
            fi
            log "Skipping incomplete so101_pick_cube at $candidate/so101_pick_cube"
        fi
        match="$(find "$candidate" -maxdepth 4 -type d -name "so101_pick_cube" 2>/dev/null | head -n 1 || true)"
        if [[ -n "$match" ]]; then
            if so101_dataset_complete "$match"; then
                copy_tree "$match" "$dst_root"
                copied=1
                break
            fi
            log "Skipping incomplete so101_pick_cube at $match"
        fi
    done

    # If direct copy didn't work, check if jsonl and videos are at the root
    if [[ "$copied" -ne 1 ]]; then
        for candidate in "${candidates[@]}"; do
            if so101_dataset_complete "$candidate"; then
                copy_tree "$candidate" "$dst_root"
                copied=1
                break
            fi
        done
    fi

    if [[ "$copied" -ne 1 ]]; then
        die "Could not find SO101 dataset in $src_dir. Expected jsonl/ and videos/ directories."
    fi
    if ! so101_dataset_complete "$dst_root"; then
        die "$dst_root is incomplete. Expected both jsonl/ and videos/."
    fi

    log "Organized SO101 data under $dst_root"
}

download_dataset() {
    DATA_ROOT="$(abspath "$DATA_ROOT")"
    DATASET_DOWNLOAD_DIR="$(abspath "$DATASET_DOWNLOAD_DIR")"

    if dir_has_entries "$DATASET_DOWNLOAD_DIR" && [[ "$FORCE" -ne 1 ]]; then
        log "Using existing dataset download dir $DATASET_DOWNLOAD_DIR (use --force to download again)"
    else
        log "Downloading dataset $DATASET_REPO to $DATASET_DOWNLOAD_DIR"
        hf_download dataset "$DATASET_REPO" "$DATASET_DOWNLOAD_DIR"
    fi
    extract_archives "$DATASET_DOWNLOAD_DIR"
    organize_so101_dataset "$DATASET_DOWNLOAD_DIR" "$DATA_ROOT"
}

download_model() {
    MODEL_DIR="$(abspath "$MODEL_DIR")"
    if dir_has_entries "$MODEL_DIR" && [[ "$FORCE" -ne 1 ]]; then
        log "Using existing model dir $MODEL_DIR (use --force to download again)"
        return 0
    fi

    if [[ -z "$MODEL_REPO" ]]; then
        die "No pretrained model repo was provided. Pass --model-repo or set PRETRAIN_MODEL_REPO, or pre-populate --model-dir."
    fi

    log "Downloading model $MODEL_REPO to $MODEL_DIR"
    hf_download model "$MODEL_REPO" "$MODEL_DIR"
}

train_exp() {
    log "Launching SO101 LoRA SFT: $PROJECT_ROOT/playground/dm05_so101_lora.py"
    exec "$SCRIPT_DIR/dm05_launcher.sh" \
        --exp "$PROJECT_ROOT/playground/dm05_so101_lora.py" \
        --nproc_per_node "$NPROC_PER_NODE" \
        -- \
        --task train \
        "${TRAIN_ARGS[@]}"
}

parse_args() {
    if [[ "$#" -gt 0 ]]; then
        case "$1" in
            all|dataset|model|train)
                COMMAND="$1"
                shift
                ;;
        esac
    fi

    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --dataset-repo)
                DATASET_REPO="$2"
                shift 2
                ;;
            --data-root)
                DATA_ROOT="$2"
                shift 2
                ;;
            --dataset-download-dir)
                DATASET_DOWNLOAD_DIR="$2"
                shift 2
                ;;
            --model-repo)
                MODEL_REPO="$2"
                shift 2
                ;;
            --model-dir)
                MODEL_DIR="$2"
                shift 2
                ;;
            --nproc-per-node|--nproc_per_node)
                NPROC_PER_NODE="$2"
                shift 2
                ;;
            --force)
                FORCE=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --)
                shift
                TRAIN_ARGS+=("$@")
                break
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
    done
}

main() {
    parse_args "$@"
    if [[ -z "$COMMAND" ]]; then
        usage >&2
        die "Subcommand is required: all, dataset, model, or train."
    fi

    case "$COMMAND" in
        dataset)
            download_dataset
            ;;
        model)
            download_model
            ;;
        train)
            train_exp
            ;;
        all)
            download_dataset
            download_model
            train_exp
            ;;
        *)
            die "Unknown command: $COMMAND"
            ;;
    esac
}

main "$@"
