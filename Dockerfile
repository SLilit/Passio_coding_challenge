FROM tensorflow/tensorflow:latest

RUN python3 --version
RUN pip3 --version
    
COPY create_model.py /challenge/

WORKDIR /challenge

CMD ["python3", "create_model.py"]
