import scipy.misc as spm

INFILE = 'data/3DUS/np/np06resized.npy'
MASKFILE = 'data/3DUS/np/mask.png'

mask = spm.imread(MASKFILE)[:,:,1]


