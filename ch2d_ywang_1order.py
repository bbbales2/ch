import matplotlib.pyplot as plt
import numpy
import os
import math
import scipy.integrate
import skimage.io, skimage.util, skimage.filters

def rescale(signal, minimum, maximum):
    mins = signal.min()
    maxs = signal.max()
    
    output = (maximum - minimum) * (signal - mins) / (maxs - mins) + minimum

    return output

print "Setting up constants"

signal = numpy.zeros((256, 256)) # This is the size of your system

dx = 1

alpha = 10.0
y = 5.0

Wx = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[1], 1.0)
Wy = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[0], 1.0)
wx, wy = numpy.meshgrid(Wx, Wy)
wx2, wy2 = numpy.meshgrid(Wx * Wx, Wy * Wy)

wx = wx
wy = wy
wx2 = wx2
wy2 = wy2

c1 = 0.125
c2 = 0.383

c11 = 2.31
c12 = 1.09
c44 = 1.57
eps0 = None

xi = (c11 - c12 - 2 * c44) / c44

sigmas = numpy.ndarray(wx.shape)

e1 = numpy.ndarray(wx.shape)
e2 = numpy.ndarray(wx.shape)

ws = numpy.array([wx, wy])

norms = numpy.linalg.norm(ws, 2, axis = 0)

e1 = ws[0] / norms
e2 = ws[1] / norms

def sig(e1, e2):
    sigmas = (1 + \
       2 * xi * (e1 * e1 * e2 * e2)) / \
   ((1 + \
     xi * ((c11 + c12) / c11) * (e1 * e1 * e2 * e2)))

    return sigmas

sigmas = sig(e1, e2)
sigmas[0, 0] = 1

print "Adjusting to V..."

def asig(angles):
    out = numpy.ndarray(len(angles))
    for i, a in enumerate(angles):
        out[i] = sig(math.cos(a), math.sin(a))
        
    return out

meanSigma = scipy.integrate.quadrature(asig, 0.0, 2 * numpy.pi, tol = 1e-12, rtol = 1e-12, maxiter = 100)

V = sigmas - meanSigma[0] / (2 * numpy.pi)

print "Running code..."

b0 = 905.387
b2 = 211.611
b3 = -165.031
b4 = 135.637

c = scipy.fftpack.fft2(0.10 + (0.24 - 0.125) * numpy.random.rand(signal.shape[0], signal.shape[1]))
n = scipy.fftpack.fft2((numpy.random.rand(signal.shape[0], signal.shape[1])))

dt = 0.5e-3

sig1 = 5.0
sig2 = -50.0

chi = 0.4
mu = 1773.0

def dfdc(c, n):
    return b0 * (c - c1) - 0.5 * b2 * n **2

def dfdn(c, n):
    return b2 * (c2 - c) * n + b3 * n**2 + b4 * n**3

for t in range(0, 1001):
    c0 = numpy.array(c)
    n0 = numpy.array(n)

    c0real = numpy.real(scipy.fftpack.ifft2(c0))
    n0real = numpy.real(scipy.fftpack.ifft2(n0))

    dfdn_ = scipy.fftpack.fft2(dfdn(c0real, n0real))
    dfdc_ = scipy.fftpack.fft2(dfdc(c0real, n0real))

    real = numpy.real(scipy.fftpack.ifft2(c))
    if (t) % 100 == 0 or t == 0:
        plt.title("t = {0}".format(t))
        real = numpy.real(scipy.fftpack.ifft2(c))
        plt.subplot(1, 2, 1)
        plt.imshow(real, cmap = plt.cm.Greys)
        plt.title('c')
        plt.colorbar(fraction=0.046, pad=0.04)
        real = numpy.real(scipy.fftpack.ifft2(n))
        plt.subplot(1, 2, 2)
        plt.imshow(real, cmap = plt.cm.Greys)
        plt.title('n')
        plt.colorbar(fraction=0.046, pad=0.04)
        fig = plt.gcf()
        fig.set_size_inches(24,15)
        plt.show()
        
    n = n0 - dt * (sig1 * (wx2 + wy2) * n0 + dfdn_)

    mc = mu * V * c0
    innerx = scipy.fftpack.fft2(scipy.fftpack.ifft2(1j * wx * (-mc + dfdc_ - sig2 * (wx2 + wy2) * c0)) * c0real * (1 - c0real))
    innery = scipy.fftpack.fft2(scipy.fftpack.ifft2(1j * wy * (-mc + dfdc_ - sig2 * (wx2 + wy2) * c0)) * c0real * (1 - c0real))
    c = c0 + 1j * dt * chi * (innerx * wx + innery * wy)

