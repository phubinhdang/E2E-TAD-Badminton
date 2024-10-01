# no python 3.9 support by nvidia, so we take 3.8 as cuda111 supports 3.8
# and cuda111 is required by the cuda extension in models/ops
# see https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel-23-04
# see https://download.pytorch.org/whl/torch_stable.html, cp39 mean python 3.9
FROM nvcr.io/nvidia/tensorrt:23.04-py3
# cv2 depends on libgl1: https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update
RUN apt-get install git libgl1 -y

WORKDIR /tad
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
