import numpy as np
from matplotlib import pyplot as plt
import scipy.signal

def avg_filter(data, resolution):
    out = []
    coeff = 1.0/resolution
    for a in range(len(data)):
        count = 0
        for i in range(resolution):
            if a - i >= 0:
                count += coeff * data[a - i]
        out.append(count)
    return out

def fir(data, imp):
    out = [] 
    for a in range(len(data)):
        count = 0
        for i in range(len(imp)):
            if a - i >= 0:
                count += imp[i] * data[a - i]
        out.append(count)
    return out

def gauss_imp(resolution, B):
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * B)
    imp = np.arange(-resolution/2, resolution/2)
    imp = imp / float(sigma)
    imp = np.power(imp, 2)
    imp = imp * -0.5
    imp = np.exp(imp)
    imp = np.sqrt(2.0 * np.pi) * B / np.sqrt(np.log(2.0)) * imp
    return imp

def run_avg():
    x_axis = np.arange(60, step=0.1)
    orig = np.sin(x_axis)
    
    noise = np.random.normal(0, 0.5, 600)
    
    noisy = orig + noise
    avg = avg_filter(noisy, 10)

    plt.figure()
    plt.plot(x_axis, orig, color='blue')
    #plt.plot(x_axis, noisy, color='red')
    #plt.plot(x_axis, avg, color='black')
    
    orig_fft = np.abs(np.fft.rfft(orig))
    noisy_fft = np.fft.fft(noisy)
    avg_fft = np.fft.fft(avg)
    
    plt.figure()
    plt.plot(orig_fft, color='blue')
    #plt.plot(x_axis, noisy_fft, color='red')
    #plt.plot(x_axis, avg_fft, color='black')
    plt.show()

def run_gauss():
    x_axis = np.arange(20)
    rand = np.random.randint(2, size=20).tolist()
    rand = scipy.signal.square(x_axis)
    data = []
    for d in rand:
        data += [d] * 16
    
    plt.figure()
    
    data = np.array(data)
    data[np.where(data == -1)] = 0
    plt.plot(data, marker='d', color='black', drawstyle='steps-pre')
    
    imp = gauss_imp(50, 0.05)
    data[np.where(data == 0)] = -1 
    out = fir(data, imp)
    plt.plot(out, marker='d', color='red')
    plt.savefig('data.png')

    plt.figure()
    data_fft = np.abs(np.fft.fftshift(np.fft.fft(data)))
    out_fft = np.abs(np.fft.fftshift(np.fft.fft(out)))
    plt.plot(data_fft, marker='d', color='black')
    plt.plot(out_fft, marker='d', color='red')
    plt.savefig('data_fft.png')

    plt.figure() 
    gauss_fft = np.abs(np.fft.fftshift(np.fft.fft(imp)))
    plt.plot(gauss_fft, marker='d', color='blue')
    plt.savefig('gaussian_filter.png')

    plt.show()

def run_box():
    x_axis = np.linspace(0, 0.005, 100, endpoint=False)
    rand = scipy.signal.square(2 * np.pi * 1000 * x_axis)
    data = rand
    
    data_fft = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data_fft), 0.005/100)
    shifted = np.fft.fftshift(data_fft)
    data_fft_centered = np.abs(shifted)

    frequencies = np.array(sorted(frequencies))
    plt.figure()
    plt.xlabel('Frequency (Hz)')
    plt.title('FFT')
    
    orig, = plt.plot(frequencies, data_fft_centered, marker='d', color='red', label='Input FFT')
    data_fft_centered[np.where(frequencies > 1000)] = 0
    data_fft_centered[np.where(frequencies < -1000)] = 0
    filt, = plt.plot(sorted(frequencies), data_fft_centered, marker='d', color='blue', label='Filtered FFT')
    plt.legend([orig, filt])
    
    shifted[np.where(frequencies > 1000)] = 0
    shifted[np.where(frequencies < -1000)] = 0
    unshifted = np.fft.ifftshift(shifted)
    filtered_data = np.fft.ifft(unshifted)
    
    plt.figure()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Data')
    inp, = plt.plot(x_axis, rand, marker='.', color='red', label='Square Wave Input Data')
    filt, = plt.plot(x_axis, filtered_data, marker='.', color='blue', label='Filtered Data')
    plt.legend([inp, filt])
    plt.show()
run_gauss()
