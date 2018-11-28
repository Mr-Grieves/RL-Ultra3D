import numpy as np
import scipy.ndimage as spm
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN

'''
NOTES:
 - Extent/Bounds = Spacing
'''

dataset_num = '06'
filename = 'VTK/VTK'+dataset_num+'.vtk'
out_file = 'np/np'+dataset_num+'resized'

reader = vtkStructuredPointsReader()
reader.SetFileName(filename)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()

data = reader.GetOutput()
dim = data.GetDimensions()
vec = list(dim)
vec = [i for i in dim]
print(vec)

# Will likely need this at some point:
'''spacing = data.GetSpacing()
print('Spacing',spacing)
save(METAFILE,spacing)'''

points = VN.vtk_to_numpy(data.GetPointData().GetArray('scalars'))

print('max =',max(points),'\tmin =',min(points))
psum = sum(points)

print('Before resize:')
points = points.reshape(vec,order='C')
plen = np.prod(points.shape)
print('shape =',points.shape)
print('mean =',(psum/plen))

print('After resize:')
points = spm.zoom(points,[1,208/224,1]) # Resize y dim down to 208
plen = np.prod(points.shape)
print('shape =',points.shape)
print('mean =',(psum/plen))

np.save(out_file,points)
print('Saved dataset',dataset_num,'to',out_file,'.npy')