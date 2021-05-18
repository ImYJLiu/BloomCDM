import torch
print(torch.__version__)
prob = torch.tensor([0.3,0.4,0.6,0.7])

out = (prob>0.5).float()

print(out)