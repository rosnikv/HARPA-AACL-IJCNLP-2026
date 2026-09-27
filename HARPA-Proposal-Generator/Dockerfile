# Use a base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set environment variables for consistent locale behavior
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Set the working directory inside the container
WORKDIR /stage/

# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -r requirements.txt
RUN pip install python-dotenv

# Pre-download Hugging Face model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the entire project into the container
COPY . .

# No default command — lets you choose at runtime
# You can run anything like: docker run my-image python evaluation/run_eval.py
