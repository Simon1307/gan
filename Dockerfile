FROM python:3.8

COPY requirements.txt /requirements.txt

RUN pip install --requirement requirements.txt

ADD src /src
ADD static /static
ADD templates /templates
ADD artifacts /artifacts
ADD app.py app.py

ENTRYPOINT ["python"]
CMD ["app.py"]