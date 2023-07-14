# Get nvidia's pytorch image
FROM nvcr.io/nvidia/pytorch:22.05-py3

# Change working directory
WORKDIR /workspace/screen_detection_cnn

# Install dependencies from requirements.txt and additional libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sets the container to run jupyter lab on start
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''","--NotebookApp.password=''"]
