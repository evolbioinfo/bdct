FROM python:3.10-slim

RUN mkdir /pasteur

# Install bdct
RUN cd /usr/local/ && pip3 install --no-cache-dir bdct==0.1.26

# The entrypoint runs bdct_infer with command line arguments
ENTRYPOINT ["bdct_infer"]