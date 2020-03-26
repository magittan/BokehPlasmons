import os,h5py
import numpy as np
from common.baseclasses import ArrayWithAxes as AWA

def Progress(i,L,last):
    next = last+10
    percent = 100*i/L
    if percent >= next:
        print('{}% complete...'.format(next))
        return next
    else:
        return last

def load_eigpairs(basedir=os.path.dirname("./"),eigpair_fname="UnitSquareMesh_100x100_1000_eigenbasis.h5"):
    """Normalization by sum always ensures that integration will be like summing, which is
    much simpler than keeping track of dx, dy..."""

    global eigpairs
    eigpairs = dict()

    path=os.path.join(basedir,eigpair_fname)

    with h5py.File(path,'r') as f:
        for key in list(f.keys()):
            eigfunc=np.array(f.get(key))
            eigfunc/=np.sqrt(np.sum(np.abs(eigfunc)**2))
            eigpairs[float(key)] = AWA(eigfunc,\
                                       axes=[np.linspace(0,1,eigfunc.shape[0]),\
                                             np.linspace(0,1,eigfunc.shape[1])])
    return eigpairs
