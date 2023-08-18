FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04
LABEL authors="yuan.feng"

# Install system libraries required by OpenCV.
RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY main.py requirements.txt ./
COPY script /app/script
RUN pip install -r requirements.txt
RUN pip uninstall transformer-engine -y


ENTRYPOINT ["python","./main.py"]
