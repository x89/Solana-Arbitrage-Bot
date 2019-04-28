iFROM python:alpine

COPY . /shreddit
WORKDIR /shreddit
RUN pip install -r requirements.txt
RUN python setup.py install

VOLUME /config
WORKDIR /config
