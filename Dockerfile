#Grab the latest alpine image
FROM ubuntu

# Install python and pip
RUN apt-get update
RUN apt-get -y upgrade
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y python3 python3-pip bash
RUN apt-get install -y python3-opencv
ADD ./webapp/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip3 install -r /tmp/requirements.txt

# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

RUN adduser myuser

# ENV HOME /root
# WORKDIR /root

RUN pip3 install gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT mainRecognition:app 