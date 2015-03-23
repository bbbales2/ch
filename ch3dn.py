import numpy
import os
import math
import scipy.integrate
import mahotas
import skimage.io, skimage.util, skimage.filters
import pycuda, pycuda.compiler, scikits.cuda.fft

print "Loading files"

def saveStack(ims, fname):
    try:
        os.mkdir(os.path.dirname(fname))
    except:
        pass
    
    for i in range(ims.shape[-1]):
        skimage.io.imsave('{0}_{1:0>3}.png'.format(fname, i), ims[:, :, i])

ims = skimage.io.imread_collection('/home/bbales2/virt/ch/gpu2/out0/*.png')
ims = ims.concatenate()[0:36]
ims = numpy.rollaxis(ims, 0, 3)

ns = skimage.io.imread_collection('/home/bbales2/virt/ch/input/*.png')
ns = ns.concatenate()[0:36]
ns = numpy.rollaxis(ns, 0, 3)

ims2 = []
for i in range(ims.shape[2]):
    ims2.append(scipy.misc.imresize(ims[:, :, i], 1.0))
    
ims2 = numpy.array(ims2)
ims2 = numpy.rollaxis(ims2, 0, 3)
print ims2.shape
#ims2 = numpy.array(ims)

def rescale(signal, minimum, maximum):
    mins = signal.min()
    maxs = signal.max()
    
    output = (maximum - minimum) * (signal - mins) / (maxs - mins) + minimum
    threshold = skimage.filters.threshold_otsu(output)
    #print threshold
    
    #output -= threshold
    return output

print "Setting up constants"
pr = 'float32'
signal = rescale(ims2, 0.0, 1.0).astype(pr)

Wx = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[0], 1.0).astype(pr)
Wy = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[1], 1.0).astype(pr)
Wz = 2.0 * numpy.pi * numpy.fft.fftfreq(signal.shape[2], 20.0 / 13.35144043).astype(pr)#
wx, wy, wz = numpy.meshgrid(Wx, Wy, Wz)
wx = wx.astype(pr)
wy = wy.astype(pr)
wz = wz.astype(pr)

wx2, wy2, wz2 = numpy.meshgrid(Wx * Wx, Wy * Wy, Wz * Wz)
wx2 = wx2.astype(pr)
wy2 = wy2.astype(pr)
wz2 = wz2.astype(pr)


c1 = numpy.float32(0.125)
c2 = numpy.float32(0.383)

b0 = numpy.float32(905.387)
b3 = numpy.float32(-165.031)
b4 = numpy.float32((100.0) / 3.0)
b4p = numpy.float32((35.637) / 9.0)
#b2 = 211.611
b2 = numpy.float32((-(1.0 / 3.0) * b3 -b4 -3 * b4p) / (c2 - 0.24))

c11 = numpy.float32(2.31)
c12 = numpy.float32(1.09)
c44 = numpy.float32(1.57) # * 1e12 erg/cm^3
eps0 = None

xi = (c11 - c12 - 2 * c44) / c44

#def sig(k):
#    km = numpy.linalg.norm(k, 2)
    
#    if km < 1e-12:
#        return 1.0
    
#    e = k / km
    
#    return c11 * (1 + 2 * xi * (e[0]**2 * e[1]**2)) / (c11 + xi * (c11 + c12) * (e[0]**2 * e[1]**2))

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

print "Adjusting to V..."

def asig(theta, phi):
    x = numpy.sin(theta) * numpy.cos(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(theta)
            
    out = sig(x, y, z)
        
    return numpy.sin(theta) * out

meanSigma = scipy.integrate.dblquad(asig, 0.0, 2 * numpy.pi, lambda x : 0.0, lambda x : numpy.pi)

print meanSigma[0] / (4.0 * numpy.pi), meanSigma[1]

V = sigmas - meanSigma[0] / (4.0 * numpy.pi)###numpy.mean(sigmas.flatten())#
#V[0, 0, 0] = 1
#print meanSigma

print "Building Cuda..."

pycuda.driver.init()
context = pycuda.tools.make_default_context()

#def dfdc(c, n):
#    return b0 * (c - c1) - 0.5 * b2 * n * n

#def dfdn(c, n):
#    return b2 * (c2 - c) * n + b3 * n * n + b4 * n * n * n

def dfdc(c, n1, n2, n3):
    return b0 * (c - c1) - 0.5 * b2 * (n1 ** 2 + n2 ** 2 + n3 ** 2)

def dfdn1(c, n1, n2, n3):
    return b2 * (c2 - c) * n1 + (1.0 / 3.0) * b3 * n2 * n3 + b4 * n1**3 + b4p * (n1**2 + n2**2 + n3**2) * n1

def dfdn2(c, n1, n2, n3):
    return b2 * (c2 - c) * n2 + (1.0 / 3.0) * b3 * n1 * n3 + b4 * n2**3 + b4p * (n1**2 + n2**2 + n3**2) * n2

def dfdn3(c, n1, n2, n3):
    return b2 * (c2 - c) * n3 + (1.0 / 3.0) * b3 * n1 * n2 + b4 * n3**3 + b4p * (n1**2 + n2**2 + n3**2) * n3

mod = pycuda.compiler.SourceModule("""
#include <cuComplex.h>

__global__ void dfdc(float scaleFact, float b0, float c1, float b2, cuComplex *c, cuComplex *n1, cuComplex *n2, cuComplex *n3, cuComplex *z)
{
  const int i = ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x) + threadIdx.x;
  float cv = scaleFact * c[i].x;
  float nv1 = scaleFact * n1[i].x;
  float nv2 = scaleFact * n2[i].x;
  float nv3 = scaleFact * n3[i].x;

  z[i] = make_cuFloatComplex(b0 * (cv - c1) - 0.5 * b2 * (nv1 * nv1 + nv2 * nv2 + nv3 * nv3), 0.0);
}

__global__ void dfdn(float scaleFact, float b2, float c2, float b3, float b4, float b4p, cuComplex *c, cuComplex *n1, cuComplex *n2, cuComplex *n3, cuComplex *z)
{
//
  const int i = ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x) + threadIdx.x;

  float cv = scaleFact * c[i].x; 
  float nv1 = scaleFact * n1[i].x; 
  float nv2 = scaleFact * n2[i].x; 
  float nv3 = scaleFact * n3[i].x; 

  z[i] = make_cuFloatComplex(b2 * (c2 - cv) * nv1 + (1.0 / 3.0) * b3 * nv2 * nv3 + b4 * nv1 * nv1 * nv1 + b4p * (nv1 * nv1 + nv2 * nv2 + nv3 * nv3) * nv1, 0.0);
}

__global__ void nupdate(float dt, float sig1, float *wx, float *wy, float *wz, cuComplex *dfdn, cuComplex *n)
{
  const int i = ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x) + threadIdx.x;
  
  float ws2 = wx[i] * wx[i] + wy[i] * wy[i] + wz[i] * wz[i];
  cuComplex tmp1 = make_cuFloatComplex(ws2 * n[i].x, ws2 * n[i].y);
  cuComplex tmp2 = cuCaddf(make_cuFloatComplex(tmp1.x * sig1, tmp1.y * sig1), dfdn[i]);

  //n0 - dt * (sig1 * (wx2 + wy2 + wz2) * n0 + dfdn_)
  n[i] = cuCsubf(n[i], make_cuFloatComplex(tmp2.x * dt, tmp2.y * dt));
}
//fde = (-mu * V * c0 + dfdc_ - sig2 * (wx2 + wy2 + wz2) * c0)
//mu, sig2, V, c0, wx2, wy2, wz2, innerx, innery, innerz
__global__ void fdeupdate(float mu, float sig2,
                cuComplex *V, cuComplex *c,
                float *wx, float *wy, float *wz,
                cuComplex *dfdc,
                cuComplex *innerx, cuComplex *innery, cuComplex *innerz)
{
  const int i = ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x) + threadIdx.x;
  
  float ws2 = wx[i] * wx[i] + wy[i] * wy[i] + wz[i] * wz[i];
  cuComplex tmp1 = make_cuFloatComplex(ws2 * c[i].x, ws2 * c[i].y);
  cuComplex tmp2 = cuCaddf(make_cuFloatComplex(-tmp1.x * sig2, -tmp1.y * sig2), dfdc[i]);
  cuComplex tmp3 = cuCmulf(V[i], c[i]);
  cuComplex tmp4 = cuCaddf(make_cuFloatComplex(-tmp3.x * mu, -tmp3.y * mu), tmp2);
  
  cuComplex tmpx0 = make_cuFloatComplex(wx[i] * tmp4.x, wx[i] * tmp4.y);
  cuComplex tmpy0 = make_cuFloatComplex(wy[i] * tmp4.x, wy[i] * tmp4.y);
  cuComplex tmpz0 = make_cuFloatComplex(wz[i] * tmp4.x, wz[i] * tmp4.y);
  
  innerx[i] = make_cuFloatComplex(-tmpx0.y, tmpx0.x);
  innery[i] = make_cuFloatComplex(-tmpy0.y, tmpy0.x);
  innerz[i] = make_cuFloatComplex(-tmpz0.y, tmpz0.x);
}

//innerx * c0real * (1 - c0real)
__global__ void c0update(float scaleFact, cuComplex *inner, cuComplex *c0real)
{
  const int i = ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x) + threadIdx.x;
   
  float cf = scaleFact * c0real[i].x * (1.0 - c0real[i].x * scaleFact);
  inner[i] = make_cuFloatComplex(scaleFact * inner[i].x * cf, scaleFact * inner[i].y * cf);
}

//c = c0 + 1j * dt * chi * (innerx * wx + innery * wy + innerz * wz)
//dt, chi, innerx, wx, innery, wy, innerz, wz
__global__ void cupdate(float dt, float chi,
                cuComplex *innerx, float *wx,
                cuComplex *innery, float *wy,
                cuComplex *innerz, float *wz,
                cuComplex *c)
{
  const int i = ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x) + threadIdx.x;
  
  cuComplex tmp0 = make_cuFloatComplex(innerx[i].x * wx[i], innerx[i].y * wx[i]),
      tmp1 = make_cuFloatComplex(innery[i].x * wy[i], innery[i].y * wy[i]),
      tmp2 = make_cuFloatComplex(innerz[i].x * wz[i], innerz[i].y * wz[i]);
      
  cuComplex tmp3 = make_cuFloatComplex(tmp0.x + tmp1.x + tmp2.x, tmp0.y + tmp1.y + tmp2.y);
  
  c[i] = cuCaddf(make_cuFloatComplex(-tmp3.y * dt * chi, tmp3.x * dt * chi), c[i]);
}
""")

dfdc = mod.get_function("dfdc")
dfdn = mod.get_function("dfdn")
nupdate = mod.get_function("nupdate")
c0update = mod.get_function("c0update")
fdeupdate = mod.get_function("fdeupdate")
cupdate = mod.get_function("cupdate")

c = scipy.fftpack.fftn(rescale(signal, 0.125, 0.24))
n1 = numpy.zeros(signal.shape).astype('complex64')
n2 = numpy.zeros(signal.shape).astype('complex64')
n3 = numpy.zeros(signal.shape).astype('complex64')

options = sorted(list(set(ns.flatten())))

lookup = dict([(opt, i) for i, opt in enumerate(options)])
factors = [[0, 0, 0], [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]

for i in range(0, signal.shape[0]):
    for j in range(0, signal.shape[1]):
        for k in range(0, signal.shape[2]):
            n1v, n2v, n3v = factors[lookup[ns[i, j, k]]]
        
            n1[i, j, k] = n1v
            n2[i, j, k] = n2v
            n3[i, j, k] = n3v

n1 = scipy.fftpack.fftn(n1)
n2 = scipy.fftpack.fftn(n2)
n3 = scipy.fftpack.fftn(n3)

dt = numpy.float32(0.5e-3)

sig1 = numpy.float32(5.0)
sig2 = numpy.float32(-50.0)

chi = numpy.float32(0.4)
mu = numpy.float32(1273.0)

c0real = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)

dfdn1_ = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)
dfdn2_ = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)
dfdn3_ = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)
dfdc_ = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)

innerx = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)
innery = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)
innerz = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)

n1real = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)
n2real = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)
n3real = pycuda.gpuarray.GPUArray(signal.shape, dtype=numpy.complex64)

gV = pycuda.gpuarray.to_gpu(V.astype('complex64'))

gwx = pycuda.gpuarray.to_gpu(wx)
gwy = pycuda.gpuarray.to_gpu(wy)
gwz = pycuda.gpuarray.to_gpu(wz)

gc = pycuda.gpuarray.to_gpu(c)
gn1 = pycuda.gpuarray.to_gpu(n1)
gn2 = pycuda.gpuarray.to_gpu(n2)
gn3 = pycuda.gpuarray.to_gpu(n3)
tmp2 = pycuda.gpuarray.to_gpu(n1)

plan = scikits.cuda.fft.Plan(signal.shape, numpy.complex64, numpy.complex64)

threads = (64, 1, 1)
blockDim = (signal.shape[0] * signal.shape[1] / threads[0], signal.shape[2], 1)
print blockDim, threads

saveStack(rescale(numpy.real(numpy.fft.ifftn(c)), 0.0, 1.0), 'gpu/in')
scaleFact = numpy.float32(1.0 / (signal.shape[0] * signal.shape[1] * signal.shape[2]))

import time
print "Starting execution"
starttime = time.time()
steps = 1001
for t in range(0, steps):
    scikits.cuda.fft.ifft(gc, c0real, plan)
    scikits.cuda.fft.ifft(gn1, n1real, plan)
    scikits.cuda.fft.ifft(gn2, n2real, plan)
    scikits.cuda.fft.ifft(gn3, n3real, plan)
    #c0real = numpy.real(scipy.fftpack.ifftn(c0))
    #n0real = numpy.real(scipy.fftpack.ifftn(n0))
    dfdn(scaleFact, b2, c2, b3, b4, b4p, c0real, n1real, n2real, n3real, dfdn1_, block = threads, grid = blockDim)
    dfdn(scaleFact, b2, c2, b3, b4, b4p, c0real, n2real, n1real, n3real, dfdn2_, block = threads, grid = blockDim)
    dfdn(scaleFact, b2, c2, b3, b4, b4p, c0real, n3real, n1real, n2real, dfdn3_, block = threads, grid = blockDim)
    dfdc(scaleFact, b0, c1, b2, c0real, n1real, n2real, n3real, dfdc_, block = threads, grid = blockDim)
    #print numpy.imag(dfdc_.get()[1, 1, 0:10])
    #print numpy.fft.ifftn(gc.get())[1, 0, 0]
    #print b2, c2, b3, b4, numpy.real(dfdn_.get())[1, 0, 0], numpy.real(c0real.get())[1, 0, 0], numpy.real(n0real.get())[1, 0, 0]
    scikits.cuda.fft.fft(dfdn1_, dfdn1_, plan)
    scikits.cuda.fft.fft(dfdn2_, dfdn2_, plan)
    scikits.cuda.fft.fft(dfdn3_, dfdn3_, plan)
    scikits.cuda.fft.fft(dfdc_, dfdc_, plan)

    if t % 50 == 0:
        scikits.cuda.fft.ifft(gc, tmp2, plan)
        c = rescale(numpy.real(tmp2.get()), 0.0, 1.0)
        saveStack(c, 'gpu4/out{0}/out{0}'.format(t))

        scikits.cuda.fft.ifft(gn1, tmp2, plan)
        c = rescale(numpy.abs(numpy.real(tmp2.get())), 0.0, 1.0)
        threshold = skimage.filters.threshold_otsu(c)
        out = c >= threshold
        saveStack(out * 1.0, 'gpu5/seg{0}/out{0}'.format(t))

        #mark = mahotas.labeled.borders((c >= threshold))

        #scikits.cuda.fft.ifft(gn2, tmp2, plan)
        #c = rescale(numpy.real(tmp2.get()), 0.0, 1.0)
        #saveStack(c, 'gpu4/outn2{0}/out{0}'.format(t))

        #scikits.cuda.fft.ifft(gn3, tmp2, plan)
        #c = rescale(numpy.real(tmp2.get()), 0.0, 1.0)
        #saveStack(c, 'gpu4/outn3{0}/out{0}'.format(t))

    #print gc.get()[:5, 0, 0]
    #print gn.get()[:4, 0, 0]
    #print '----------'
    #n = n0 - dt * (sig1 * (wx2 + wy2 + wz2) * n0 + dfdn_)
    nupdate(dt, sig1, gwx, gwy, gwz, dfdn1_, gn1, block = threads, grid = blockDim)
    nupdate(dt, sig1, gwx, gwy, gwz, dfdn2_, gn2, block = threads, grid = blockDim)
    nupdate(dt, sig1, gwx, gwy, gwz, dfdn3_, gn3, block = threads, grid = blockDim)
    fdeupdate(mu, sig2, gV, gc, gwx, gwy, gwz, dfdc_, innerx, innery, innerz, block = threads, grid = blockDim)

    #print gV.get()[1, 1, 1], gc.get()[1, 1, 1], gwx.get()[1, 1, 1], gwy.get()[1, 1, 1], gwz.get()[1, 1, 1]
    scikits.cuda.fft.ifft(innerx, innerx, plan)
    scikits.cuda.fft.ifft(innery, innery, plan)
    scikits.cuda.fft.ifft(innerz, innerz, plan)
    c0update(scaleFact, innerx, c0real, block = threads, grid = blockDim)
    c0update(scaleFact, innery, c0real, block = threads, grid = blockDim)
    c0update(scaleFact, innerz, c0real, block = threads, grid = blockDim)
    scikits.cuda.fft.fft(innerx, innerx, plan)
    scikits.cuda.fft.fft(innery, innery, plan)
    scikits.cuda.fft.fft(innerz, innerz, plan)
    #print innerx.get()[1, 1, 1]
    #print innery.get()[1, 1, 1]
    #print innerz.get()[1, 1, 1]
    
    #fde = (-mu * V * c0 + dfdc_ - sig2 * (wx2 + wy2 + wz2) * c0)
    #innerx = scipy.fftpack.fftn(scipy.fftpack.ifftn(1j * wx * fde) * c0real * (1 - c0real))
    #innery = scipy.fftpack.fftn(scipy.fftpack.ifftn(1j * wy * fde) * c0real * (1 - c0real))
    #innerz = scipy.fftpack.fftn(scipy.fftpack.ifftn(1j * wz * fde) * c0real * (1 - c0real))
    cupdate(dt, chi, innerx, gwx, innery, gwy, innerz, gwz, gc, block = threads, grid = blockDim)
    #c = c0 + 1j * dt * chi * (innerx * wx + innery * wy + innerz * wz)
    #print gn.get()[1, 1, 1]
print "Time per cycle: ", (time.time() - starttime) / float(steps)

scikits.cuda.fft.ifft(gn1, tmp2, plan)
c = rescale(numpy.abs(numpy.real(tmp2.get())), 0.0, 1.0)

threshold = skimage.filters.threshold_otsu(c)
out = c >= threshold
saveStack(out * 1.0, 'gpu5/seg{0}/seg{0}'.format(t))
mark = mahotas.labeled.borders((c >= threshold))

context.pop()
