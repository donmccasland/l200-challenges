FROM python:3.9

EXPOSE 8081
ENV PORT 8081

WORKDIR /home

COPY . /home
RUN pip install -r /home/requirements.txt

CMD python3 /home/main.py
