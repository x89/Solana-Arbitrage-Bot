FROM python:alpine

RUN mkdir -p /config
VOLUME /config

ENV CONFIG_FILE shreddit.yaml

COPY . /shreddit
WORKDIR /shreddit

RUN pip install -r requirements.txt
RUN pip install .

WORKDIR /config

CMD ["/bin/sh", "-c", "/usr/local/bin/shreddit -c ${CONFIG_FILE}"]
