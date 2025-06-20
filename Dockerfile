FROM python:3.8-slim-buster

RUN apt update && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

CMD [ "python3", "app.py" ]