FROM telestoai/model-api-base:latest

# Install all your dependencies here, preferably using just one RUN command
RUN python -m pip --no-cache-dir install \
    torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    python -m pip --no-cache-dir install -U pillow==6.*

# model.py contains a class that extends ClassificationModelBase from telesto/model.py
COPY *.py /
COPY model.pt /

# The default values for the model Python module and the model weights path can be overwritten here
ENV MODEL_CLASS model.ExampleClassificationModel
ENV MODEL_PATH /model.pt
