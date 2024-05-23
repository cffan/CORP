FROM tensorflow/tensorflow:2.7.0-gpu

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt update
RUN apt -y install apt-transport-https ca-certificates gnupg software-properties-common wget python3.9 python3-pip python3.9-dev
RUN python3.9 -m pip install --upgrade pip

# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt update
RUN apt -y install cmake

# Install pytorch
RUN pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Download CORP
RUN wget https://codeload.github.com/cffan/CORP/zip/refs/heads/falcon -O CORP.zip
RUN unzip CORP.zip
RUN rm -rf CORP.zip

# Install LM decoder
WORKDIR "/CORP-falcon/LanguageModelDecoder/runtime/server/x86"
RUN python3.9 setup.py install

# Install Neural decoder
WORKDIR "/CORP-falcon/NeuralDecoder"
RUN pip install -e .

# Install Online recalibrator
WORKDIR "/CORP-falcon/OnlineRecalibrator"
RUN pip install -e .

# Install Falcon challenge
RUN pip install falcon_challenge --upgrade
# WORKDIR /
# RUN wget https://codeload.github.com/snel-repo/falcon-challenge/zip/refs/heads/h2_eval -O falcon.zip
# RUN unzip falcon.zip
# RUN rm -rf falcon.zip
# WORKDIR "/falcon-challenge-h2_eval"
<<<<<<< HEAD
# RUN pip install -e .
=======

ENV EVAL_DATA_PATH "/dataset/evaluation_data"
ENV SPLIT "h2"
ENV PHASE "test"
ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH = "/tmp/ground_truth.pkl"
>>>>>>> 3aa849e (Update for batch mode)

# Add pretrained models, data, and codes
WORKDIR /
COPY ./ /falcon_challenge
WORKDIR /falcon_challenge/falcon-challenge/
RUN pip install .

# Run decoder
WORKDIR /falcon_challenge
<<<<<<< HEAD
CMD ["/bin/bash", "-c", "python3.9 run_evalulator.py --evaluation local --corp-config config/falcon_held_out_oracle.yaml --phase test"]
=======
CMD ["/bin/bash", "-c", "python3.9 run_evalulator.py --evaluation local --corp-config config/falcon_held_out_eval.yaml --phase ${PHASE} --split ${SPLIT}"]
>>>>>>> 3aa849e (Update for batch mode)
