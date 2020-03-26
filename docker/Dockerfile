FROM continuumio/anaconda3

USER root

RUN apt-get -y update && apt install -y build-essential

COPY spec-list.pathflow.txt .

RUN conda create --name default --file spec-list.pathflow.txt && \
  rm spec-list.pathflow.txt

RUN apt-get install -y openslide-tools

COPY requirements.txt .

RUN pip install -r requirements.txt && \
    rm requirements.txt

RUN pip install git+https://github.com/jlevy44/PathFlowAI.git

RUN install_apex

#ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
