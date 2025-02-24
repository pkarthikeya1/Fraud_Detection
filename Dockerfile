FROM python:3.10
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN pip install -r requirements.txt
CMD python src/frontend.py