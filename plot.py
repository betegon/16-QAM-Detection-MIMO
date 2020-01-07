import matplotlib.pyplot as plt
import numpy as np


k = 2  # Antennas
snr_array = np.array([8, 10, 12, 14, 16, 18])

quantiz = []
eigen = []
random = []

ser_quantiz = []
ser_eigen = []
ser_random = []
log_errors = np.loadtxt('mosek-2antenas.txt', delimiter=',')
for i in range(len(snr_array)):
    err1 = log_errors[:, i*3]
    err2 = log_errors[:, (i*3)+1]
    err3 = log_errors[:, (i*3)+2]
    ser_quantiz.append((np.sum(err1))/(len(log_errors[:, 0])*(2*k)))
    ser_eigen.append((np.sum(err2))/(len(log_errors[:, 0])*(2*k)))
    ser_random.append((np.sum(err3))/(len(log_errors[:, 0])*(2*k)))

fig, ax = plt.subplots()
ax.plot(snr_array, ser_quantiz, marker='o', label='Simple quantization', color=(0.694, 0.09412, 0.1765))  # 177 G: 24 B: 45
ax.plot(snr_array, ser_eigen, marker='v', label='Eigenvalue decomposition', color=(0.65, 0.65, 0.65), linestyle=':')
ax.plot(snr_array, ser_random, marker='s', label='randomization', color=(0, 0.439216, 0.4706))
ax.spines['bottom'].set_color((0, 0, 0, 0.69))
ax.spines['top'].set_color((0, 0, 0, 0.69))
ax.xaxis.label.set_color((0, 0, 0, 0.69))

ax.spines['right'].set_color((0, 0, 0, 0.69))
ax.spines['left'].set_color((0, 0, 0, 0.69))
ax.yaxis.label.set_color((0, 0, 0, 0.69))
ax.tick_params(axis='both', colors=(0, 0, 0, 0.69))
plt.yscale('log')
plt.ylim([10**-4, 1])
plt.xlim([snr_array[0], snr_array[-1]])
plt.grid(which='major', axis='y', color='#999999', linestyle='--', linewidth=0.4)
plt.grid(which='major', axis='x', color='#999999', linestyle='--', alpha=0.4, linewidth=0.4)
plt.minorticks_on()
plt.grid(b=True, which='minor', axis='y', color='#999999', linestyle='--', alpha=0.2, linewidth=0.5)
ax.set_xlabel(r'$SNR[dB]$', fontsize='large')
ax.set_ylabel('Symbol Error Rate', fontsize='large')
legend = plt.legend(loc='lower left', fontsize='large')
plt.setp(legend.get_texts(), color=(0, 0, 0, 0.69))
plt.title('16-QAM, K={} Users'.format(k), fontsize='x-large', color=(0, 0, 0, 0.69))
plt.show()
