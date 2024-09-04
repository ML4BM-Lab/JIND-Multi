
# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Create a new Conda environment and install Python 3.7.16
RUN conda create -n jind python=3.7.16 --yes

# Activate the Conda environment and install pip and dependencies
RUN /bin/bash -c "source activate jind && conda install pip --yes"

# Install the package in editable mode
RUN /bin/bash -c "source activate jind && pip install -e /app/."

# Activate the Conda environment and run the application
CMD ["/bin/bash", "-c", "source activate jind && exec /bin/bash"]

