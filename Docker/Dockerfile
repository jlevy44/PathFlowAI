FROM continuumio/anaconda3

USER root

RUN apt-get -y update && apt install -y build-essential

COPY spec-list.pathflow.txt .

RUN conda create --name default --file spec-list.pathflow.txt && \
  rm spec-list.pathflow.txt 

COPY req-list.pathflow.txt .

RUN pip install -r req-list.pathflow.txt && \
    rm req-list.pathflow.txt

RUN pip install git+https://github.com/jlevy44/PathFlowAI.git@0ad5e1025a9b82c489b9a854e14580df914d8d13

RUN install_apex

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
