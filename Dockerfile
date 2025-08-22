FROM ubuntu:latest
LABEL authors="saatarko"

ENTRYPOINT ["top", "-b"]