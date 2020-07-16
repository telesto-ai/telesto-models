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

# Copying files to the image
COPY model /

# Installing dependencies
RUN python -m pip --no-cache-dir install -r requirements.txt
