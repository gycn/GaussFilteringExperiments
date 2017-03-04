import numpy as np
from matplotlib import pyplot as plt 
import scipy.signal

def gauss_imp(BT, span, sample_rate, frequency): 
  #num_coefficients is the sampling rate of the filter
  #3db_bandwidth is the 3db bandwidth of the gaussian filter
  #symbol time is the time difference between two symbols during transmission
  alph = np.sqrt(2. * np.pi / np.log(2)) * BT
  imp = np.linspace(- span * sample_rate / 2, span * sample_rate / 2, span*sample_rate)
  #generate coefficents
  imp *= np.sqrt(np.pi) * alph / sample_rate
  imp = - np.power(imp, 2)
  imp = np.exp(imp)
  imp *= alph
  imp /= sum(imp) 
  #plot gaussian filter
  plt.figure()
  x_axis = np.arange(- span * sample_rate / 2, span * sample_rate / 2)
  plt.plot(x_axis, imp, marker='d')
  plt.title('Gaussian Filter Coefficients') 
  plt.xlabel('Filter Position')
  plt.ylabel('Response')
  plt.savefig('gauss_filter_coeff.png')
  
  #FFT of filter coefficients
  filter_fft = np.abs(np.fft.fftshift(np.fft.fft(imp))) #FFT of filter
  filter_fft_db = 20 * np.log10(filter_fft)

  #plot filter fft
  plt.figure()
  x_axis = np.arange(-len(filter_fft)/2, len(filter_fft)/2) / sample_rate
  plt.plot(x_axis, filter_fft_db)
  plt.title('Gaussian Filter FFT')
  plt.ylabel('Response(db)')
  plt.xlabel('Frequency(Hz)')
  plt.savefig('filter_fft.png')
  
  imp /= sum(imp) #normalize
  return imp 

def fir(data, imp):
	#Convolve data with imp
  out = []
  for a in range(len(data)):
      count = 0
      for i in range(len(imp)):
          if a - i >= 0:
              count += imp[i] * data[a - i]
      out.append(count)
  return out

def generate_periodic_data(num_symbols, num_samples, frequency):
  #generate square wave periodic
  return scipy.signal.square(np.linspace(0, num_symbols * num_samples/(2 * frequency * num_samples), num_symbols * num_samples, endpoint=False) * 2 * np.pi * frequency)

def run_gauss(data, imp, frequency, sample_rate):
  x_axis = np.arange(len(data)) / (2.0 * frequency * sample_rate)
  
  #generate random square wave
  #rand = np.random.randint(2, size=8).tolist()
  #data = []
  #for i in rand:
  #  data += [i] * 8
  #data = np.array(data)
 
  #Set 0's in data to -1
  data[np.where(data == 0)] = -1
  
  #Convolve gaussian coefficients with sampled data
  filtered = fir(data, imp) 
  
  #plot data & filtered data
  plt.figure() 
  unfilt, = plt.plot(x_axis, data, marker='d', color='red', label='Data')
  filt, = plt.plot(x_axis, filtered, marker='d', color='blue', label='Filtered Data')
  plt.legend([unfilt, filt])
  plt.title('Time Domain')
  plt.xlabel('Time(s)')	
  plt.ylabel('Amplitude')	
  plt.savefig('data.png')

  #Run FFT 
  data_fft = np.abs(np.fft.fftshift(np.fft.fft(data))) #FFT of original data
  out_fft = np.abs(np.fft.fftshift(np.fft.fft(filtered))) #FFT of filtered data
 
  #Get x-axis for FFT plots
  data_frequencies = sorted(np.fft.fftfreq(len(data), 1/(2.0 * sample_rate * frequency)))
  
  #Plot Data FFT
  plt.figure()
  data_f,  = plt.plot(data_frequencies, data_fft, color='red', label='Data FFT')
  out_f,   = plt.plot(data_frequencies, out_fft, color='blue', label='Filtered Data FFT')
  plt.legend([data_f, out_f])
  plt.xlabel('Frequency(Hz)')
  plt.ylabel('Response')
  plt.title('Frequency Domain')
  plt.savefig('data_fft.png')

  plt.show()

if __name__ == '__main__':
  num_symbols = 8 #number of symbols in data
  num_samples = 8 #number of samples for each symbol 
  freq = 1000 #frequency of the square wave
  gauss_span = 4 #how many symbols the gaussian filter should be applied to
  BT = 0.5 #3 db bandwidth of the gaussian filter in units of bit rate

  #generate periodic square wave data
  data = generate_periodic_data(num_symbols, num_samples, freq)
  #generate gaussian filter coefficients
  gauss_filter = gauss_imp(BT, gauss_span, num_samples, freq)
  
  run_gauss(data, gauss_filter, freq, num_samples)
  #show all plots
  plt.show()
