FROM telestoai/model-api-base:latest

# base_models.py contains a class that extends ClassificationModelBase from telesto/base_models.py
COPY *.py /
COPY model.pt /
# requirements.txt contains the Python packages required to run predictions
COPY requirements.txt /

# Install all your dependencies here, preferably using just one RUN command
RUN python -m pip --no-cache-dir install -r requirements.txt
