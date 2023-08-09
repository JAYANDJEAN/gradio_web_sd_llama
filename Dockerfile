FROM nvcr.io/nvidia/pytorch:23.06-py3
LABEL authors="yuan.feng"

RUN mkdir /app
COPY * /app
WORKDIR /app
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python","./app_llama.py"]
