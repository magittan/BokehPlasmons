from TipModeling import SampleResponse, TipResponse
from common.baseclasses import ArrayWithAxes as AWA
from Utils import load_eigpairs, planewave
import matplotlib.pyplot as plt
import numpy as np

def SimpleTest(q=44, _N_sample_eigenbasis=100, _N_tip_eigenbasis=1):
    eigpairs = load_eigpairs(basedir="../sample_eigenbasis_data")

    Sample = SampleResponse(eigpairs,qw=q,N_sample_eigenbasis=_N_sample_eigenbasis)
    Tip = TipResponse(Sample.xs,Sample.ys,q=q,N_tip_eigenbasis=_N_tip_eigenbasis)

    d = Sample.RasterScan(Tip)

    plt.figure()
    plt.imshow(np.abs(d['P'])); plt.title('P');plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(d['R'])); plt.title('R');plt.colorbar()
    plt.show()

def RibbonTest(_N_sample_eigenbasis=100, _N_tip_eigenbasis=1):
    show_eigs = False
    run_test = True
    xs,ys = np.arange(101), np.arange(101); L=xs.max()
    xv,yv = np.meshgrid(xs,ys)
    eigpairs = {}

    graphene_ribbon=False
    if graphene_ribbon:
        q0=np.pi/L #This is for particle in box (allowed wavelength is n*2*L)
        for n in range(1,_N_sample_eigenbasis+1):
            qx = n*q0
            pw = AWA(planewave(qx,0,xv,yv,x0=0,phi0=np.pi/2), axes = [xs,ys]) #cosine waves, this is for
            eigpairs[qx**2]=pw/np.sqrt(np.sum(pw**2))

    else:

        q0=2*np.pi/L #This is for infinite sample
        for n in range(1,_N_sample_eigenbasis+1):
            qx = n*q0
            pw = AWA(planewave(qx,0,xv,yv,x0=0,phi0=np.pi/2), axes = [xs,ys]) #cosine waves
            eigpairs[qx**2]=pw/np.sqrt(np.sum(pw**2))

            pw2 = AWA(planewave(qx,0,xv,yv,x0=0,phi0=0), axes = [xs,ys]) #sine waves, This is for infinite sample
            eigpairs[qx**2+1e-9]=pw2/np.sqrt(np.sum(pw2**2)) #This is for infinite sample

    if show_eigs:
        for i,q in enumerate(list(eigpairs.keys())):
            if i<5:
                plt.figure()
                plt.imshow(eigpairs[q])
                plt.title("q={}".format(q))
        plt.show()

    if run_test:
        q=2*np.pi/L*20
        Sample = SampleResponse(eigpairs,qw=q,N_sample_eigenbasis=_N_sample_eigenbasis)
        Tip = TipResponse(Sample.xs, Sample.ys,q=q,N_tip_eigenbasis=_N_tip_eigenbasis)

        d = Sample.RasterScan(Tip)
        plt.figure()
        plt.imshow(np.abs(d['P'])); plt.title('P');plt.colorbar()
        plt.figure()
        plt.imshow(np.abs(d['R'])); plt.title('R');plt.colorbar()
        plt.show()

def main():
    TESTS = {
        "simple": SimpleTest,
        "ribbon": RibbonTest
    }
    test = "ribbon"
    TESTS[test](_N_sample_eigenbasis=100, _N_tip_eigenbasis=3)

if __name__=="__main__": main()
