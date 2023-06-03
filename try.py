import torch
model = torch.load('model.pth')
model.eval()
print(model(torch.tensor([0,0,0])))