FROM python:alpine

COPY . /shreddit
WORKDIR /shreddit
RUN pip install -r requirements.txt && python setup.py install

CMD ["/usr/local/bin/shreddit"]
