FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Ubuntu 22.04 mirror setup (supports both sources.list and sources.list.d formats)
RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i -e "s/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && \
        sed -i -e "s/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list; \
    elif [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i -e "s|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g" /etc/apt/sources.list.d/ubuntu.sources && \
        sed -i -e "s|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g" /etc/apt/sources.list.d/ubuntu.sources; \
    fi && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && apt-get update -y && \
    apt-get install --assume-yes --fix-missing build-essential && \
    apt-get install -y openssh-server vim git curl tmux git-lfs && \
    apt-get install -y libibverbs1 librdmacm1 && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_25.5.1-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    /opt/conda/bin/conda config --set show_channel_urls yes && \
    pip config set global.index-url https://mirrors.ivolces.com/pypi/simple/

RUN /opt/conda/bin/conda create -n opendm python=3.10 -y

# Install PyTorch with CUDA 12.8 support, then install the project
RUN /bin/bash -c "source activate opendm && \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"

# Install flash-attn (requires ninja and packaging)
RUN /bin/bash -c "source activate opendm && \
        pip install ninja packaging && \
        MAX_JOBS=2 pip install flash-attn --no-build-isolation"

COPY . /app/opendm/
WORKDIR /app/opendm/

# Install both the base package and the fast-inference dependency layer so the
# image can run either backend without extra container-local setup.
RUN /bin/bash -c "source activate opendm && \
        pip install -e . && \
        pip install -e '.[fast-infer]'"

RUN /opt/conda/bin/conda init bash

CMD ["bash"]
