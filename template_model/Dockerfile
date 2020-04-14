FROM telestoai/model-api-base:latest

# Install all your dependencies here, preferably using just one RUN command
RUN python -m pip --no-cache-dir install \

# copy all necessary files to /
# model.py contains a class that extends ClassificationModelBase from telesto/model.py
COPY *.py /
COPY model.pt /

# The default values for the model Python module and the model weights path can be overwritten here
ENV MODEL_CLASS model.ClassificationModel
ENV MODEL_PATH /model.pt
