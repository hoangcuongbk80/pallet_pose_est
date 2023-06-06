FROM nvcr.io/nvidia/pytorch:20.12-py3

# Install linux packages
RUN apt-get update && apt-get install -y --no-install-recommends screen libgl1-mesa-glx libglfw3-dev libgles2-mesa-dev

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip3 install -r requirements.txt

# Create working directory in image container
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy all contents to image container
COPY . /usr/src/app
