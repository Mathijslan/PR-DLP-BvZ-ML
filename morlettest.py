import matplotlib.pyplot as plt
import numpy as np

# TIME DOMAIN
w = 20
t = np.arange(0.01, w*2+0.01, 0.01)

s = 8
tmp1 = (t-w)/s
sigma = 0.16
tmp2 = np.exp(-(tmp1**2)/(2.0*sigma**2))
psi = (1/s)*np.pi**(-1/4)*np.cos(2*np.pi*tmp1)*tmp2

plt.subplot(2,1,1)
plt.plot(t,psi)
plt.plot([0,w*2],[np.exp(-2),np.exp(-2)])
plt.plot([0,w*2],[-np.exp(-2),-np.exp(-2)])
plt.title('Morlet time-domain')

#FREQUENCY DOMAIN
w0 = 2*np.pi
x = np.arange(0, 10.01, 0.01)
fourier_wavelength = 4*np.pi / (w0 + np.sqrt(2 + w0**2))

tmp1 = s*(x/fourier_wavelength)*sigma - w0*sigma
tmp1 = -(tmp1**2)/2.0
psi = np.exp(tmp1)

plt.subplot(2,1,2)
plt.plot(x,psi)
plt.xlim([0, 2*np.pi])
plt.title('Morlet frequency-domain')


print(np.geomspace(0.2, 100, 5))
plt.show()

accs = np.load('EMODB_MODELS/sigma_range_test/sigma_accuries22-1.npy', allow_pickle=True)

print(accs)