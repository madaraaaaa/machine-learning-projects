FROM python:3.8.12-slim

#RUN pip install pipenv
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    liblapack-dev


WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt



#COPY ["Pipfile", "Pipfile.lock", "./"]

# RUN pipenv install --deploy --system



COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["python", "predict.py"]
