import matplotlib.pyplot as plt
import numpy
import math
import scipy
import mahotas
import skimage.io, skimage.util, skimage.filters
import time
import os

def saveStack(ims, fname):
    try:
        print os.path.dirname(fname)
        os.mkdir(os.path.dirname(fname))
    except:
        pass
    
    for i in range(ims.shape[-1]):
        skimage.io.imsave('{0}_{1:0>3}.png'.format(fname, i), ims[:, :, i])

def rescale(signal, minimum, maximum):
    mins = signal.min()
    maxs = signal.max()
    
    output = (maximum - minimum) * (signal - mins) / (maxs - mins) + minimum

    return output

def run(initial, steps, printSteps, outdir, dx, dy, dz):
    pr = 'float64'
    signal = rescale(initial, 0.0, 1.0).astype(pr)

    try:
        os.mkdir(outdir)
    except:
        pass

    Wx = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[0], dx).astype(pr)
    Wy = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[1], dy).astype(pr)
    Wz = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[2], dz).astype(pr)#
    wx, wy, wz = numpy.meshgrid(Wx, Wy, Wz, indexing = 'ij')
    wx = wx.astype(pr)
    wy = wy.astype(pr)
    wz = wz.astype(pr)
    
    c1 = numpy.float32(0.125)
    c2 = numpy.float32(0.383)
    
    b0 = numpy.float32(905.387)
    b2 = numpy.float32(211.611)
    b3 = numpy.float32(-165.031)
    b4 = numpy.float32(135.637)
    
    c11 = numpy.float32(2.31)
    c12 = numpy.float32(1.09)
    c44 = numpy.float32(1.57) # * 1e12 erg/cm^3
    eps0 = None
    
    xi = (c11 - c12 - 2 * c44) / c44
    
    sigmas = numpy.ndarray(wx.shape).astype(pr)
    
    e1 = numpy.ndarray(wx.shape).astype(pr)
    e2 = numpy.ndarray(wx.shape).astype(pr)
    e3 = numpy.ndarray(wx.shape).astype(pr)
    
    ws = numpy.array([wx, wy, wz])
    
    norms = numpy.linalg.norm(ws, 2, axis = 0)
    
    e1 = ws[0] / norms
    e2 = ws[1] / norms
    e3 = ws[2] / norms
    
    def sig(e1, e2, e3):
        sigmas = c11 * (1 + \
                        2 * xi * (e1 * e1 * e2 * e2 + \
                                  e1 * e1 * e3 * e3 + \
                                  e2 * e2 * e3 * e3) + \
                        3 * xi * xi * e1 * e1 * e2 * e2 * e3 * e3) / \
            ((c11 + \
              xi * (c11 + c12) * (e1 * e1 * e2 * e2 + \
                                  e1 * e1 * e3 * e3 + \
                                  e2 * e2 * e3 * e3)) + \
             xi * xi * (c11 + 2 * c12 + c44) * e1 * e1 * e2 * e2 * e3 * e3)
        
        return sigmas

    sigmas = sig(e1, e2, e3)
    sigmas[0, 0, 0] = 1
    def asig(theta, phi):
        x = numpy.sin(theta) * numpy.cos(phi)
        y = numpy.sin(theta) * numpy.sin(phi)
        z = numpy.cos(theta)
        
        out = sig(x, y, z)
        
        return numpy.sin(theta) * out
        
    meanSigma = scipy.integrate.dblquad(asig, 0.0, 2 * numpy.pi, lambda x : 0.0, lambda x : numpy.pi)
    V = sigmas - meanSigma[0] / (4.0 * numpy.pi)
    
    #c = (signal.astype(pr) / 2.51) - 0.059
    #c *= signal > 0.0
    #c = c.flatten()
    c = rescale(signal, 0.12, 0.32).flatten()
    n = rescale(signal, 0.0, 1.0).flatten()

    dt = 0.5e-3
    
    sig1 = 5.0
    sig2 = -50.0
    
    chi = 0.4
    mu = 1773.0
    
    ft = 18500.0

    def dfdc(c, n):
        return b0 * (c - c1) - 0.5 * b2 * n * n

    def dfdn(c, n):
        return b2 * (c2 - c) * n + b3 * n * n + b4 * n * n * n

    print "Building diffusion matrices"
    dx = 1.0
    dy = 1.0
    dz = 1.0
    
    K1D = scipy.sparse.spdiags((numpy.ones((im.shape[0], 1)) * numpy.array([1, -2, 1])).transpose(),range(-1, 2), im.shape[0], im.shape[0])# 1d Poisson matrix
    K1D.data[1, 0] = -1
    K1D.data[1, -1] = -1
    
    I1D = scipy.sparse.eye(im.shape[1])#                       % 1d identity matrix
    K2D = scipy.sparse.kron(K1D, I1D) / dy + scipy.sparse.kron(I1D, K1D)#;            % 2d Poisson matrix
    
    I2D = scipy.sparse.eye(K2D.shape[0])
    I1D = scipy.sparse.eye(im.shape[2])
    K1D = scipy.sparse.spdiags((numpy.ones((im.shape[2], 1)) * numpy.array([1, -2, 1])).transpose(),range(-1, 2), im.shape[2], im.shape[2])# 1d Poisson matrix
    K1D.data[1, 0] = -1
    K1D.data[1, -1] = -1
    K3D = scipy.sparse.kron(K2D, I1D) / dz + scipy.sparse.kron(I2D, K1D)
    
    AN = (scipy.sparse.identity(K3D.shape[0]) - dt * sig1 * K3D).tocsr()
    AC = (scipy.sparse.identity(K3D.shape[0]) - dt * chi * sig2 * K3D.dot(K3D)).tocsr()
    
    ANdiag = AN.todia()
    ANpre = scipy.sparse.diags(1 / ANdiag.data[ANdiag.data.shape[0] / 2, :], 0)

    ACdiag = AC.todia()
    ACpre = scipy.sparse.diags(1 / ACdiag.data[ACdiag.data.shape[0] / 2, :], 0)

    solvetime = 0.0
    ffttime = 0.0
    functime = 0.0
    print "Running"
    for t in range(0, steps):
        print "t = ", t
        tmp = time.time()
        dfdn_ = dfdn(c, n)
        dfdc_ = dfdc(c, n)
        functime += time.time() - tmp
        
        if outdir:
            if (t) % printSteps == 0 or t == 0:
                print "Printing t = {0}".format(t)
                saveStack(rescale(c.reshape(im.shape), 0.0, 1.0), '{0}/c_stack{1}/im'.format(outdir, t))
                saveStack(rescale(n.reshape(im.shape), 0.0, 1.0), '{0}/n_stack{1}/im'.format(outdir, t))
                print "Printing seg"
                threshold = skimage.filters.threshold_otsu(n)
                out = n >= threshold
                saveStack(rescale(out.reshape(im.shape), 0.0, 1.0), '{0}/seg{1}/im'.format(outdir, t))
        
        b = n - dt * dfdn_
        tmp = time.time()
        n, r = scipy.sparse.linalg.cg(AN, b, x0 = b, tol = 1e-6, M = ANpre)
        #n = 2 * n - n0
        solvetime += time.time() - tmp
        tmp = time.time()
        fft = numpy.real(scipy.fftpack.ifftn(mu * V * scipy.fftpack.fftn(c.reshape(im.shape)))).flatten()
        ffttime += time.time() - tmp
        b = c + dt * chi * K3D * (dfdc_ + fft)
        tmp = time.time()
        c, r = scipy.sparse.linalg.cg(AC, b, x0 = b, tol = 1e-6, M = ACpre)
        #c = 2 * c - c0
        solvetime += time.time() - tmp
    print "Solver time {0}, FFT time {1}, energy func. evaluation time {2}".format(solvetime, ffttime, functime)

    return c, n

# Import the images
print "Importing image stack"
ims = skimage.io.imread_collection('/home/bbales2/virt/segmentation/ReneN4_FIB_Dataset/tosegment/shrunk/*.png')
#ims = skimage.io.imread_collection('/home/bbales2/virt/segmentation/UCSB_Rene88DT_FIBSS/BSEhigh/1/aligned/output/BSE1_aligned_*.png')
ims = ims.concatenate()[0:80]
ims = numpy.rollaxis(ims, 0, 3)

im = numpy.array(ims)
im = numpy.max(im.flatten()) - im

run(im, 101, 20, 'output', 1.0, 1.0, 1.0)
