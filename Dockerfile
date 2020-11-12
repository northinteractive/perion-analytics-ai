FROM ubuntu:20.04

MAINTAINER Jonny Coe "jon@northinteractive.co"

RUN apt-get update -y && apt-get install -y python3-pip python3-dev

COPY ./requirements.txt /requirements.txt

WORKDIR /

run pip3 install -r requirements.txt

COPY . /

ENTRYPOINT [ "python3" ]

CMD [ "app/app.py" ]