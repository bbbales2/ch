import numpy
import mahotas
import math
import matplotlib.pyplot as plt
import skimage.filter
import skimage.io
import skimage.util

im = skimage.io.imread('toseg2.png')
im = im[:, :, 1].reshape(im.shape[0], im.shape[1])
im = im[0:512, 0:512]

cmap = plt.cm.jet
cmap._init()
cmap._lut[:, -1] = 1
cmap._lut[0, -1] = 0

def rescale(signal):
    mins = signal.min()
    maxs = signal.max()
    
    output = 4.0 * (signal - mins) / (maxs - mins)
    threshold = skimage.filter.threshold_otsu(output)
    
    output -= threshold
    return output

signal = rescale(im)
N = signal.shape[0]

dt = 0.1
dx = 1

u = signal
u2 = numpy.fft.fft2(signal)

alpha = 0.1
y = 1.0

w = 2.0 * numpy.pi * numpy.fft.fftfreq(N, 1.0)
w2 = w * w
wx2, wy2 = numpy.meshgrid(w2, w2)
        
for t in range(0, 201):
    u3 = numpy.real(numpy.fft.ifft2(u2))
    fftlap = numpy.fft.fft2(u3 * u3 * u3)
    u2 = u2 / (1 + alpha * (wx2 + wy2) * dt * (-1 + y * (wx2 + wy2)))
    u2 = u2 - alpha * (wx2 + wy2) * dt * fftlap
    # I may have a sign wrong in the eq below, but this is the explicit version of the stuff above
    # The stuff above runs faster cause it handles the linear bits implicitly and can use a bigger timestep
    #u2 = u2 - alpha * dt * (wx2 + wy2) * (numpy.fft.fft2(u3 * u3 * u3 - u3) + y * (wx2 + wy2) * u2)

    if (t) % 100 == 0 or t == 0:
        plt.title("t = {0}".format(t))
        plt.imshow(numpy.real(numpy.fft.ifft2(u2)), cmap = plt.cm.Greys)
        plt.colorbar()
        plt.show()
     
u4 = numpy.real(numpy.fft.ifft2(u2))
threshold = skimage.filter.threshold_otsu(u4)
mark = mahotas.labeled.borders((u4 >= threshold))
plt.imshow(im, cmap = plt.cm.Greys)
plt.colorbar()
plt.imshow(mark)
fig = plt.gcf()
fig.set_size_inches(15,8)
plt.show()
