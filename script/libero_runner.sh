#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

COMMAND=""
DATASET_REPO="Dexmal/libero"
DATA_ROOT="$PROJECT_ROOT/data/libero"
DATASET_DOWNLOAD_DIR="$PROJECT_ROOT/data/.hf_downloads/libero"
MODEL_REPO="${PRETRAIN_MODEL_REPO:-Dexmal/DM05}"
MODEL_DIR="$PROJECT_ROOT/checkpoints/DM05"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
FORCE=0
TRAIN_ARGS=()

usage() {
    cat <<EOF
Usage:
  script/libero_runner.sh <all|dataset|model|train> [options] [-- train args]

Subcommand is required. Use "all" to run every step.

Commands:
  dataset    Download and organize the Hugging Face LIBERO dataset.
  model      Download the pretrained model from Hugging Face.
  train      Launch playground/dm05_libero.py.
  all        Run dataset -> model -> train.

Options:
  --dataset-repo REPO       HF dataset repo. Default: Dexmal/libero
  --data-root DIR           Final LIBERO data root; also used by training.
                            Default: ./data/libero
  --dataset-download-dir DIR
                            Temporary HF dataset download dir.
                            Default: ./data/.hf_downloads/libero
  --model-repo REPO         HF model repo. Default: Dexmal/DM05.
                            Can also use PRETRAIN_MODEL_REPO.
  --model-dir DIR           Final pretrained model dir. Default: ./checkpoints/DM05.
  --nproc-per-node N        torchrun processes per node. Default: \$NPROC_PER_NODE or 1
  --force                   Overwrite existing organized data files.
  -h, --help                Show this help.

Examples:
  script/libero_runner.sh dataset --data-root /path/to/data/libero
  script/libero_runner.sh model --model-repo org/model --model-dir /path/to/checkpoint
  script/libero_runner.sh train -- --trainer-config.num-train-steps 100
  script/libero_runner.sh all --nproc-per-node 8
EOF
}

log() {
    printf '[libero_runner] %s\n' "$*"
}

die() {
    printf '[libero_runner][error] %s\n' "$*" >&2
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

extracted_libero_complete() {
    local dir="$1"
    [[ -d "$dir/libero/libero_pi0_all/jsonl" && -d "$dir/libero/libero_pi0_all/image" ]]
}

libero_subset_complete() {
    local subset="$1"
    local dir="$2"

    case "$subset" in
        libero_pi0_all)
            [[ -d "$dir/jsonl" && -d "$dir/image" ]]
            ;;
        libero_oft_all)
            [[ -d "$dir" && -n "$(find "$dir" -mindepth 1 -maxdepth 2 2>/dev/null | head -n 1)" ]]
            ;;
        *)
            [[ -d "$dir/jsonl" && -d "$dir/video" ]]
            ;;
    esac
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
        if extracted_libero_complete "$out" && [[ "$FORCE" -ne 1 ]]; then
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

organize_libero_dataset() {
    local src_dir="$1"
    local dst_root="$2"
    local candidates=()

    mkdir -p "$dst_root"
    candidates+=("$src_dir")
    [[ -d "$src_dir/data/libero" ]] && candidates+=("$src_dir/data/libero")
    [[ -d "$src_dir/.extracted" ]] && candidates+=("$src_dir/.extracted")

    local copied=0
    local subset
    local candidate
    local match
    for subset in libero_goal libero_10 libero_spatial libero_object libero_pi0_all libero_oft_all; do
        for candidate in "${candidates[@]}"; do
            if [[ -d "$candidate/$subset" ]]; then
                if libero_subset_complete "$subset" "$candidate/$subset"; then
                    copy_tree "$candidate/$subset" "$dst_root/$subset"
                    copied=1
                    break
                fi
                log "Skipping incomplete $subset at $candidate/$subset"
            fi
            match="$(find "$candidate" -maxdepth 4 -type d -name "$subset" 2>/dev/null | head -n 1 || true)"
            if [[ -n "$match" ]]; then
                if libero_subset_complete "$subset" "$match"; then
                    copy_tree "$match" "$dst_root/$subset"
                    copied=1
                    break
                fi
                log "Skipping incomplete $subset at $match"
            fi
        done
    done

    if [[ "$copied" -ne 1 ]]; then
        die "Could not find LIBERO subset directories in $src_dir. Expected names like libero_pi0_all or libero_object."
    fi
    if [[ -d "$dst_root/libero_pi0_all" ]] && ! libero_subset_complete libero_pi0_all "$dst_root/libero_pi0_all"; then
        die "$dst_root/libero_pi0_all is incomplete. Expected both jsonl/ and image/."
    fi

    log "Organized LIBERO data under $dst_root"
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
    organize_libero_dataset "$DATASET_DOWNLOAD_DIR" "$DATA_ROOT"
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
    DATA_ROOT="$(abspath "$DATA_ROOT")"
    log "Launching LIBERO SFT: $PROJECT_ROOT/playground/dm05_libero.py"
    log "LIBERO data root: $DATA_ROOT"
    exec "$SCRIPT_DIR/dm05_launcher.sh" \
        --exp "$PROJECT_ROOT/playground/dm05_libero.py" \
        --nproc_per_node "$NPROC_PER_NODE" \
        -- \
        --task train \
        --data-config.jsonl-dir "$DATA_ROOT/libero_pi0_all" \
        --data-config.image-dir "$DATA_ROOT/libero_pi0_all/image" \
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
