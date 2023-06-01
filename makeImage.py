import torch
from vtk import *
import numpy as np

net = torch.load('model.pth')
net.eval()

resizedVol = np.zeros(shape=(150*150*150, 3))
ind = 0
for i in range(150):
    for j in range(150):
        for k in range(150):
            resizedVol[ind] = [i, j, k]
            ind = ind + 1
resizeTensor = torch.from_numpy(resizedVol)
resizeTensor = torch.tensor(resizeTensor, dtype=torch.float)



predicted_volume = net(resizeTensor)
print(predicted_volume)

ind = 0
arr = vtkFloatArray()
points = vtkPoints()


for i in range(150):
    for j in range(150):
        for k in range(150):
            cord = (i, j, k)
            arr.InsertNextValue(predicted_volume[ind])
            ind = ind + 1

imageData = vtkImageData()
imageData.SetDimensions(150, 150, 150)
imageData.GetPointData().SetScalars(arr)

writer = vtkXMLImageDataWriter()
writer.SetFileName("img.vti")
writer.SetInputData(imageData)
writer.Write()