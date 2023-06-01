from vtk import *
reader = vtkXMLImageDataReader()
reader.SetFileName('./img.vti')
reader.Update()

data = reader.GetOutput()
pressureArray = data.GetPointData().GetArray('Scalars_')
for i in range(10):
    print(pressureArray.GetValue(i))