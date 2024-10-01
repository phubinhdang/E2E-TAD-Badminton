FROM nvcr.io/nvidia/tensorrt:24.05-py3

# the project --> cv2 --> libgl1: https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update
RUN apt-get install git libgl1 -y

WORKDIR ./e2etad
COPY . .

RUN pip install -r requirements.txt