import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
#%matplotlib inline




# current directory
print(os.getcwd())

# get directories
print(os.listdir())


train_1_df = pd.read_csv('train_1.csv')
train_1_df.head()
train_1_df.shape

# fill nan values with zeros
train_1_df = pd.read_csv('train_1.csv').fillna(0)
train_1_df.info()



#------------ detect page language using re -----------------------------------

# =============================================================================
# search() vs. match()
# 
# Python offers two different primitive operations based on regular expressions: 
# re.match() checks for a match only at the beginning of the string, 
# while re.search() checks for a match anywhere in the string (this is what Perl 
# does by default).
# 
# For example:
# 
# =============================================================================
match_result = re.match("c", "abcdef")    # No match
print(match_result)
search_result =  re.search("c", "abcdef")   # Match
# <re.Match object; span=(2, 3), match='c'>


# Regular expressions beginning with '^' can be used with search() to restrict 
# the match at the beginning of the string:
re.match("c", "abcdef")    # No match
re.search("^c", "abcdef")  # No match
re.search("^a", "abcdef")  # Match
#<re.Match object; span=(0, 1), match='a'>

def get_language(page):
    re_sc_result = re.search('[a-z][a-z].wikipedia.org', page)
    
    if re_sc_result: # meaning if re_sc_result contains a value
        return re_sc_result[0][0:2]
    return 'na'

train_1_df['lang'] = train_1_df.Page.map(get_language)

from collections import Counter

print(Counter(train_1_df.lang))

train_1_df.head()
# =============================================================================
# 
# a = 2
# if a:
#     print(a)
# else:
#     print('none')
#     
# a = 2
# if (a == True):
#     print(a)
# else:
#     print('none')
# =============================================================================

lang_sets = {}

# create 8 dataframes
lang_sets['en'] = train_1_df[train_1_df.lang=='en'].iloc[:,0:-1]
lang_sets['en'].head()
lang_sets['en'].shape # Out[6]: (24108, 551)
lang_sets['en'].shape[0] # Out[5]: 24108

lang_sets['en'].sum(axis=0)

lang_sets['en'].iloc[:,1:].sum(axis=0)

# =============================================================================
# 
# lang_sets['en'].head()
# Out[38]: 
#                                                    Page     ...      2016-12-31
# 8357          !vote_en.wikipedia.org_desktop_all-agents     ...             0.0
# 8358  "Awaken,_My_Love!"_en.wikipedia.org_desktop_al...     ...          1770.0
# 8359  "European_Society_for_Clinical_Investigation"_...     ...             2.0
# 8360  "Weird_Al"_Yankovic_en.wikipedia.org_desktop_a...     ...          1098.0
# 8361     100_metres_en.wikipedia.org_desktop_all-agents     ...           272.0
# 
# [5 rows x 551 columns]
# =============================================================================

lang_sets['ja'] = train_1_df[train_1_df.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = train_1_df[train_1_df.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = train_1_df[train_1_df.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = train_1_df[train_1_df.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = train_1_df[train_1_df.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = train_1_df[train_1_df.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = train_1_df[train_1_df.lang=='es'].iloc[:,0:-1]

# lang_sets is a dictionary object
lang_sets # it contains 8 dataframe object
# each dataframe contains corresponding no. of rows x 551 columns

# =============================================================================
# 
#  145059  Resident_Evil:_Capítulo_Final_es.wikipedia.org...     ...             0.0
#  145060  Enamorándome_de_Ramón_es.wikipedia.org_all-acc...     ...             0.0
#  145061  Hasta_el_último_hombre_es.wikipedia.org_all-ac...     ...             0.0
#  145062  Francisco_el_matemático_(serie_de_televisión_d...     ...             0.0
#  
#  [14069 rows x 551 columns]}
# 
# =============================================================================

sums = {} # create a dict
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:, 1:].sum(axis=0) / lang_sets[key].shape[0]

# sums is a series dictionary which contains 8 series objects
# key is language and value is is the 
# each object contains two columns, one index column, for series, here days is the
# is the index column, and another is, sum of page views in that day
sums['en'].shape # (550,)

sums['en'].shape[0] # 550
sums['en'].shape[1] # error tuple index out of range

sums['ru'].shape[0] # 500
sums['zh'].shape[0] # 500

# read no. of rows in english dataframe
days = [i for i in range(sums['en'].shape[0])] # list of 550 values

fig = plt.figure(1,figsize=[10,10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels={'en':'English','ja':'Japanese','de':'German',
        'na':'Media','fr':'French','zh':'Chinese',
        'ru':'Russian','es':'Spanish'
       }

for key in sums:
    plt.plot(days,sums[key],label = labels[key] )
    
plt.legend()
plt.show()


# =============================================================================
# 
# English shows a much higher number of views per page, as might be expected since 
# Wikipedia is a US-based site. There is a lot more structure here than I would 
# have expected. The English and Russian plots show very large spikes around day 
# 400 (around August 2016), with several more spikes in the English data later in 
# 2016. My guess is that this is the effect of both the Summer Olympics in August 
# and the election in the US.
# 
# There's also a strange feature in the English data around day 200.
# 
# The Spanish data is very interesting too. There is a clear periodic structure there, 
# with a ~1 week fast period and what looks like a significant dip around every 6 months 
# or so.
# 
# 
# =============================================================================


# =============================================================================
# 
# There are 7 languages plus the media pages. The languages used here are: 
# English, Japanese, German, French, Chinese, Russian, and Spanish. This will make 
# any analysis of the URLs difficult since there are four different writing systems 
# to be dealt with (Latin, Cyrillic, Chinese, and Japanese). Here, I will create 
# dataframes for the different types of entries. I will then calculate the sum of 
# all views. I would note that because the data comes from several different sources, 
# the sum will likely be double counting some of the views.
# 

# =============================================================================

# =============================================================================
# 
# # Series object has only “axis 0” because it has only one dimension. 
# a_series = pd.Series(['Alif', 'Ba', 'Ta', 'Cha', 'Jim'])
# a_series[0]
# 
# b_series = pd.Series(['Ha', 'Kha', 'Daal', 'Jaal', 'Ra', 'Jha'])
# 
# ab_df = pd.DataFrame({'Column1': a_series, 'Column2': b_series})
# ab_df.head()
# 
# # Show 3rd row
# ab_df.loc[2]
# 
# ab_df.loc[2, 4] # error
# ab_df.loc[2, 'Column1'] 
# ab_df.loc[2, 'Column2']
# 
# a_series = pd.Series([1, 2, 3, 4, 5])
# b_series = pd.Series([10, 20, 30, 40, 50])
# 
# ab_df = pd.DataFrame({'Column1':a_series, 'Column2':b_series})
# ab_df.head()
# ab_df.sum(axis=0)
# 
# ab_df['Column1'].iloc[:,1:]
# ab_df['Column2'].iloc[:,1:]
# 
# c_series = pd.Series([6, 7, 8, 9, 10])
# 
# abc_df = pd.DataFrame({'Column1':a_series, 'Column2':b_series, 'Column3':c_series})
# abc_df.head()
# 
# abc_df['Column1'].head()
# abc_df['Column1'].shape
# 
# abc_df['Column1'].iloc[:, 1:].sum(axis=0) # too many indexers because there is 
# # only one column, look at the shape of both
# abc_df.iloc[:, 1:].sum(axis=0) # works
# 
# 
# df_list = abc_df.iloc[[0], :].values.tolist()
# df_list.head() # error
# df_list # 2d array
# 
# abc_df.iloc[0, :].values.tolist()
# abc_df.iloc[1, :].values.tolist()
# 
# abc_df
# abc_df.iloc[:, 1:].sum(axis=0) # works
# abc_df.iloc[:, 1:].sum(axis=1) # works
# 
# =============================================================================


# =============================================================================
# 
# Periodic Structure and FFTs
# 
# Since it looks like there is some periodic structure here, I will plot each of 
# these separately so that the scale is more visible. Along with the individual plots, 
# I will also look at the magnitude of the Fast Fourier Transform (FFT). Peaks in the 
# FFT show us the strongest frequencies in the periodic signal.
# =============================================================================



# the OP's data
# =============================================================================
# 
# Apply Fourier Transform if you find periodic data
# The problem here is that you don't have periodic data. You should always inspect 
# the data that you feed into any algorithm to make sure that it's appropriate.
# 
# =============================================================================

x = pd.read_csv('http://pastebin.com/raw.php?i=ksM4FvZS', skiprows=2, header=None).values
y = pd.read_csv('http://pastebin.com/raw.php?i=0WhjjMkb', skiprows=2, header=None).values
fig, ax = plt.subplots()
ax.plot(x, y)


from scipy.fftpack import fft

#create an array with random n numbers
x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
print(x)
#Applying the fft function
y_fft = fft(x)
print(y_fft)



time_step = 0.02
period = 5.
time_vec = np.arange(0, 20, time_step)
sig = np.sin(2 * np.pi / period * time_vec) + 0.5 *np.random.randn(time_vec.size)
print(sig.size) # 1000

# =============================================================================
# 
# We do not know the signal frequency; we only know the sampling time step of the 
# signal sig. The signal is supposed to come from a real function, so the Fourier 
# transform will be symmetric. 
# 
# The scipy.fftpack.fftfreq() function will generate the sampling frequencies and 
# scipy.fftpack.fft() will compute the fast Fourier transform.
# 
# Let us understand this with the help of an example.
# =============================================================================

# Sampling rate and time vector

from scipy import pi
start_time = 0 # seconds
end_time = 2 # seconds
sampling_rate = 1000 # Hz
N = (end_time - start_time)*sampling_rate # array size

# Frequency domain peaks
peak1_hz = 60 # Hz where the peak occurs
peak1_mag = 25 # magnitude of the peak
peak2_hz = 270 # Hz where the peak occurs
peak2_mag = 2 # magnitude of the peak

# Noise control
noise_loc = 0 # the Gaussian noise is mean-centered
noise_mag = 0.5 # magnitude of added noise

# Vibration data generation
time = np.linspace(start_time, end_time, N)
print(time)

vib_data = peak1_mag*np.sin(2*pi*peak1_hz*time) + \
           peak2_mag*np.sin(2*pi*peak2_hz*time) + \
           np.random.normal(0, noise_mag, N) 

# Data plotting
plt.plot(time[0:100], vib_data[0:100])
plt.xlabel('Time')
plt.ylabel('Vibration (g)')
plt.title('Time Domain (Healthy Machinery)');


from scipy.fftpack import fft

# Nyquist Sampling Criteria
T = 1/sampling_rate # inverse of the sampling rate
x = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

# FFT algorithm
yr = fft(vib_data) # "raw" FFT with both + and - frequencies
y = 2/N * np.abs(yr[0:np.int(N/2)]) # positive freqs only

# Plotting the results
plt.plot(x, y)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Vibration (g)')
plt.title('Frequency Domain (Healthy Machinery)');

# Time-Domain Plot
peak2_mag2 = 8 # magnitude of the peak
vib_data2 = peak1_mag*np.sin(2*pi*peak1_hz*time) + \
            peak2_mag2*np.sin(2*pi*peak2_hz*time) + \
            np.random.normal(0, noise_mag, N) 
plt.figure()
plt.plot(time[0:100], vib_data2[0:100])
plt.xlabel('Time')
plt.ylabel('Vibration (g)')
plt.title('Time Domain (Faulted Bearing)')

# Frequency-Domain Plot
yr2 = fft(vib_data2) # "raw" FFT with both + and - frequencies
y2 = 2/N * np.abs(yr2[0:np.int(N/2)]) # positive freqs only
plt.figure()
plt.plot(x, y2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Vibration (g)')
plt.title('Frequency Domain (Faulted Bearing)');


#-----------

from scipy.fftpack import fft
def plot_with_fft(key):

    fig = plt.figure(1,figsize=[15,5])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.title(labels[key])
    plt.plot(days,sums[key],label = labels[key] )
    
    fig = plt.figure(2,figsize=[12,5])
    fft_complex = fft(sums[key])
    fft_mag = [np.sqrt(np.real(x)*np.real(x)+np.imag(x)*np.imag(x)) for x in fft_complex]
    fft_xvals = [day / days[-1] for day in days]
    npts = len(fft_xvals) // 2 + 1
    fft_mag = fft_mag[:npts]
    fft_xvals = fft_xvals[:npts]
        
    plt.ylabel('FFT Magnitude')
    plt.xlabel(r"Frequency [days]$^{-1}$")
    plt.title('Fourier Transform')
    plt.plot(fft_xvals[1:],fft_mag[1:],label = labels[key] )
    # Draw lines at 1, 1/2, and 1/3 week periods
    plt.axvline(x=1./7,color='red',alpha=0.3)
    plt.axvline(x=2./7,color='red',alpha=0.3)
    plt.axvline(x=3./7,color='red',alpha=0.3)

    plt.show()

for key in sums:
    plot_with_fft(key)
    

# =============================================================================
#     
# From this we see that while the Spanish data has the strongest periodic features, 
# most of the other languages show some periodicity as well. For some reason the 
# Russian and media data do not seem to show much. I plotted red lines where a 
# period of 1, 1/2, and 1/3 week would appear. We see that the periodic features 
# are mainly at 1 and 1/2 week. This is not surprising since browsing habits may 
# differ on weekdays compared to weekends, leading to peaks in the FFTs at frequencies 
# of n/(1 week) for integer n. So, we've learned now that page views are not at all 
# smooth. There is some regular variation from day to day, but there are also large 
# effects that can happen quite suddenly. A model likely will not be able to predict 
# the sudden spikes unless it can be fed more information about what is going on in 
# the world that day.
# =============================================================================


def plot_entry(key,idx):
    data = lang_sets[key].iloc[idx,1:]
    fig = plt.figure(1,figsize=(10,5))
    plt.plot(days,data)
    plt.xlabel('day')
    plt.ylabel('views')
    plt.title(train_1_df.iloc[lang_sets[key].index[idx],0])
    
    plt.show()
    
    
idx = [1, 5, 10, 50, 100, 250,500, 750,1000,1500,2000,3000,4000,5000]
for i in idx:
    plot_entry('en',i)    