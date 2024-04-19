import torch


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("CUDA is available:", torch.cuda.is_available())  # Print para verificar si CUDA está disponible