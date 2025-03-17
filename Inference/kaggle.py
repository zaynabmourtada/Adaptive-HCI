import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("vaibhao/handwritten-characters")

print("Path to dataset files:", path)
