import numpy as np
import matplotlib.pyplot as plt

def message_signal(t):
    if 1 <= t < 1.9:
        return -t + 2
    elif 1.9 <= t < 2:
        return 0.1
    else:
        return 0
    
fc = 25 #Hz
Ac = 1 #Volt

# Carculate and Plot the Message Signal m(t)

t = np.arange(1,2,1/500)
mt = np.array([message_signal(t1) for t1 in t])
plt.plot(t,mt)
plt.grid()
plt.show()

###########################

# Carculate and Plot the Carrier Signal

c = Ac*np.cos(2*np.pi*fc*t)
plt.plot(t,c)
plt.grid()
plt.show()

############################

# Calculate and Plot the DSC - AM - SC Signal

dsb = np.multiply(mt,c)
plt.plot(t,dsb)
plt.grid()
plt.show()

#############################

# Average Power

av_power = np.average(Ac**2*(np.abs(mt)**2)/4)
print(av_power)

###########################

# Define the parameters
sampling_rate = 1000  # Sampling rate (samples per second)
duration = 2  # Duration of the signal in seconds

# Generate the time axis
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Calculate the message signal
mt = np.array([message_signal(t1) for t1 in t])

# Calculate the Fourier transform and power spectral density
fft = np.fft.fft(mt)
freq = np.fft.fftfreq(len(t), 1/sampling_rate)
psd = np.abs(fft)**2

# Plot the power spectral density
plt.plot(freq, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Spectral Density of Message Signal')
plt.grid(True)
plt.show()

##############################################

sampling_rate = 1000  # Sampling rate (samples per second)
duration = 2  # Duration of the signal in seconds

# Generate the time axis
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Calculate the carrier signal
c = Ac * np.cos(2 * np.pi * fc * t)

# Calculate the Fourier transform and power spectral density
fft = np.fft.fft(c)
freq = np.fft.fftfreq(len(t), 1/sampling_rate)
psd = np.abs(fft)**2

# Plot the power spectral density
plt.plot(freq, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Spectral Density of Carrier Signal')
plt.grid(True)
plt.xlim([-40, 40])
plt.show()

##############################################

# Define the parameters
sampling_rate = 1000  # Sampling rate (samples per second)
duration = 2  # Duration of the signal in seconds

# Generate the time axis
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Calculate the message signal
mt = np.array([message_signal(t1) for t1 in t])

# Calculate the carrier signal
c = Ac * np.cos(2 * np.pi * fc * t)

# Calculate the DSB-AM-SC signal
dsb = np.multiply(mt, c)

# Calculate the Fourier transform and power spectral density
fft = np.fft.fft(dsb)
freq = np.fft.fftfreq(len(dsb), 1/sampling_rate)
psd = np.abs(fft)**2

# Plot the power spectral density
plt.plot(freq, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Spectral Density of DSB-AM-SC Signal')
plt.grid(True)
plt.xlim([-100, 100])
plt.show()

##############################

snr_db = 10
av_power_db = 10 * np.log10(av_power)
noise_avg_db = av_power_db - snr_db
noise_avg_lin = 10 ** (noise_avg_db / 10)
m_watts = mt ** 2
noise = np.random.normal(0,np.sqrt(noise_avg_lin), len(m_watts))
mt_noise = mt + noise

# Define the parameters
sampling_rate = 1000  # Sampling rate (samples per second)
duration = 2  # Duration of the signal in seconds

# Generate the time axis
t = np.linspace(1, duration, int(sampling_rate * duration), endpoint=False)

# Calculate the message signal
mt= np.array([message_signal(t1) for t1 in t])
mt_noise = mt + noise

# Calculate the carrier signal
c = Ac * np.cos(2 * np.pi * fc * t)

# Calculate the DSB-AM-SC signal
dsb = np.multiply(mt, c)

r = dsb + noise

# Calculate the Fourier transform and power spectral density
fft = np.fft.fft(r)
freq = np.fft.fftfreq(len(r), 1/sampling_rate)
psd = np.abs(fft)**2

# Plot the power spectral density
plt.plot(freq, psd)
plt.grid()
plt.title("Message Signal Spectrum with Noise")
plt.xlim([-50, 50])
plt.ylim([0, 5000])
plt.show()

plt.plot(t, mt_noise)
plt.show()

#The code was made by Panagiota Fasili and Omouridou Eleni

