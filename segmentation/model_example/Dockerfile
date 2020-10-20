##################################################################
##################################################################
##                                                              ##
## Dockerfile for telesto.ai template submission.               ##
## !!!NOTE!!!                                                   ##
## Please don't modify this Dockerfile. For security reasons    ##
## we don't allow custom Dockerfiles for model submissions.     ##
##                                                              ##
##################################################################
##################################################################

FROM telestoai/model-api-base:latest

WORKDIR /root

# Installing dependencies
COPY requirements.txt .
RUN python -m pip --no-cache-dir install -r requirements.txt

# Copying files to the image
COPY . .

CMD [ "segmentation" ]
