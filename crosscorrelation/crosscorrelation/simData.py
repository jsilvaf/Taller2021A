"""
OVRO 40m program variability
 
Routines to generate all kind of mock data sets for training of analysis
methods. This includes sysnthetic data sets, bootstrap and monte carlo 
analysis of light curves.

Walter Max-Moerbeck, 16 April 2010
"""

import scipy
from scipy import array, random, concatenate, arange, zeros, fftpack, absolute,\
    real, imag, ceil, floor, argmin, sqrt, sum, var, mean, log2, log, unique,\
    size, where

import pylab

import time
import sys

#-------------------------------------------------------------------------------
# 1/f^a noise generation
#-------------------------------------------------------------------------------
def oofNoise(N,
             tSamp=1.0,
             beta=1.0,
             real_imag_ratio=1.0e9):
    """ Generates a time series with 1/f^beta power spectrum.
    Uses the algorithm described on Timmer and Kronig 1995.

    Input
    -----
    N      Number of time series samples. Even number for simplicity 
    tsamp  Sampling period for time series
    beta   Exponent of power spectrum P ~ 1 / f^beta
    real_imag_ratio   Tolerance for imaginary part

    Output
    ------
    t, s   Time series. Time on same units as tSamp
    
    Notes
    -----
    - This particular code generates a simple power psd. I will write a function
    to generate general psds if needed in the future.

    Walter Max-Moerbeck, August 04, 2009
    """

    #--------------------------------------------------
    # Verify input, only works for N even
    if N % 2 == 1:
        print 'N has to be an even number'
        return array([]), array([])

    #--------------------------------------------------
    # Fourier components of time series

    # Random Fourier components for positive frequencies.
    ft_r_p = random.normal(0, 1.0, N/2 + 1)
    ft_i_p = random.normal(0, 1.0, N/2 + 1)

    # Make the average a real number
    ft_i_p[0] = 0.0

    # FT(fNyquist) is real for N even, as in this code
    ft_i_p[-1] = 0.0

    # Fourier components for negative values
    # Follow the conventions for frequency order of fft and ifft
    ft_r = concatenate((ft_r_p, ft_r_p[N/2 - 1:0:-1]))
    ft_i = concatenate((ft_i_p, -ft_i_p[N/2 - 1:0:-1]))
    FT = ft_r + 1j * ft_i

    # Calculate frequencies and filter coefficients
    f = arange(N/2 + 1) / (N * tSamp)
    f = concatenate((f, f[N/2 - 1:0:-1]))

    # Zero frequency component is made zero
    H = zeros(N)
    H[0] = 0.0
    H[1:] = f[1:] ** (-beta / 2.0)

    # Give it the actual power law shape
    FT = FT * H

    #--------------------------------------------------
    # Take the inverse Fourier transform to get time series
    s = fftpack.ifft(FT)
    t = tSamp * arange(N)

    #--------------------------------------------------
    # Check that s is positive, if imaginary part is to large give a warning
    r = max(absolute(real(s) / imag(s)))
    if r < real_imag_ratio:
        print 'Real/Imag ratio is smaller than tolerance ', real_imag_ratio

    # Parseval
    # print sum(absolute(FT)**2) / N**2, var(real(s), ddof=0.0)

    return t, real(s)


#-------------------------------------------------------------------------------
def simulateLightCurveSampling(t,
                               tSamp=0.1,
                               beta=1.0,
                               real_imag_ratio=1.0e9,
                               tSim=365.0 * 20.0,
                               index_sampling=None):
    """ Simulate light curve with a given PSD exponent and sample as the input 
    light curve

    The output times are taken from the data but the samples are generated at 
    slightly different times. The difference is smaller than the samping time 
    so it might produce spourious high frequency components. I expect the effect
    not to be very large.

    input
    -----
    t                      Data set to get sampling from
    tSamp=1.0              Sampling time (I think as units in days)
    beta=1.0               Exponent of PSD (1/f^beta)
    real_imag_ratio        Upper limit for imaginary part on time series
                           See oofNoise() for details
    sigma2=1.0             Normalization for PSD fits          
    tSim                   Time lenght of simulates light curve. It has to be 
                           al least as long as t
    index_sampling=array([])        
                           Sampling pattern for light curve. It can be used when
                           using function to simulate many light curves
                           I use None to avoid modifying mutable object if I 
                           don't define default parameter on call


    output
    ------
    t               Time vector from actual data
    s_s             Simulated data with given PSD and sampled as data
    t_sim, s_sim    Finely sampled simulated light curve
    index_sampling=None  index sampling used. It speeds computation up. It has
                         to be initialized as array([]) and returns used one.

    Walter Max-Moerbeck, January 21, 2010
    """

    # number of samples required for one light curve
    # Find maximum among lenght of t and tSim
    time_length_t = t[-1] - t[0]

    if time_length_t >= tSim:

        time_length = time_length_t

    else:
        
        time_length = tSim


    # Calculate minimum number of samples needed from simulation
    N = int(ceil(( time_length / tSamp + 1)))

    # make it a power of two to improve speed of fft
    N = 2 ** int(ceil(log2(N)))

    # simulate finely sampled light curve with same time duration
    t_sim, s_sim = oofNoise(N,
                            tSamp=tSamp,
                            beta=beta,
                            real_imag_ratio=real_imag_ratio)

    # Take a section of the light curve to sample
    N_original =  int(ceil(time_length_t / tSamp + 1))
    t_sim = t_sim[:N_original]
    s_sim = s_sim[:N_original]

    # sample light curve as the data
    # make first simulated sample time equal to data time
    t_sim = t_sim - t_sim[0] + min(t)

    # Extract the samples with given sampling if given
    if len(index_sampling) == len(t):

        t_s = t[:]
        s_s = s_sim[index_sampling]

    else:
        # for each point on data time select closest simulated sample
        index_sampling = array([argmin(absolute(t_sim - t_i)) for t_i in t])
        t_s = t[:]
        s_s = s_sim[index_sampling]


    return t_s, s_s, t_sim, s_sim, index_sampling


#-------------------------------------------------------------------------------
def addObservingNoise(t, s, error,
                      t_s, s_s,
                      scale_factor=1.0):
    """ Add observational noise to simulated light curve

    input
    -----
    t, s, error     Reference time series, to get error from
    t_s, s_s        Time series to add error to
    scale_factor    Scale factor between reference and simulated time series

    output
    ------
    s_s             Simulated time series, with error added

    notes
    -----

    The errors used for s_s are normalized such that

    error_s_s = error_s * scale_factor

    The scale factor is computed by another routine, and is chosen to make the 
    measured PSD of the data equal to the underlaying simulated PSD, not a 
    single realization.

    The idea is basically to find a good approximation of the PSD normalization
    and use it to escale the error bars of the simulated data

    Walter Max-Moerbeck, February 14, 2011
    """

    # randomize flux with scaled error
    t_s, s_s, e_s = randomize_flux(t_s, s_s, error * scale_factor)

    return s_s


#-------------------------------------------------------------------------------
def getErrorScaleFactor(t, s, error,
                        tSamp=1.0,
                        beta=1.0,
                        tSim=0.0,
                        N_sim=100,
                        real_imag_ratio=1.e9):
    """ Simulate N_sim light curves with given PSD to get approximate
    normalization

    If there is more error than signal set scale_factor = 0.0, in which case 
    only the noise is modeled

    input
    -----
    t, s, error            Time series with error
    tSamp=1.0              Sampling period for simulated light curves
    beta=1.0               Power-law index of simulated light curves
    tSim=0.0               Lenght of simulated light curves
    N_sim=100              Number of simulated light curves
    real_imag_ratio=1.e9   Ensure small imaginary part in simulated time series

    output
    ------
    scale_factor           Factor by which the error on data has to be multiplied
                           to get same relative contribution to simulated data

    notes
    -----
    - The idea is that the variance in the data has two independent components,
    the signal and the noise. If this is the case
    
    var(data) = var(signal) + var(noise)

    This function finds the conversion factor between simulated variance, which
    is assumed to be pure signal and var(signal)

    scale_factor = sqrt(var(sim) / var(signal))



    Walter Max-Moerbeck, February 14, 2011
    """

    # Signal variance
    sigma2_data = var(s, ddof=1)
    sigma2_noise = mean(error**2)

    sigma2_signal = sigma2_data - sigma2_noise

    # Check the anomalous case when sigma2_signal <= 0.0 and set scale to 0.0
    if sigma2_signal <= 0.0:
        
        scale_factor = 0.0

        print 'sigma2_data = %f, sigma2_signal = %f, sigma2_noise = %f' %\
            (sigma2_data,
             sigma2_signal,
             sigma2_noise)

        sys.exit("getErrorScaleFactor() error. sigma2_signal <= 0.0")

        return scale_factor


    # Simulate light curve and get sigma2 for each one
    SIGMA2_SIM = []

    # Index sampling array to speed up computation
    index_sampling = array([])

    for i in xrange(N_sim):

        # Simulate light curve with given sampling
        t_s, s_s, t_sim, s_sim, index_sampling =\
            simulateLightCurveSampling(t,
                                       tSamp=tSamp,
                                       beta=beta,
                                       real_imag_ratio=real_imag_ratio,
                                       tSim=tSim,
                                       index_sampling=index_sampling)
        
        # Get sigma^2 for simulation with given sampling
        SIGMA2_SIM.append(var(s_s))
        
    # Compute scale factor from the mean sigma2 for simulations and data
    sigma2_mean_sim = mean(SIGMA2_SIM)

    # Get scale factor for error bars on simulated data
    # error_s_s = scale_factor * error_s
    scale_factor = sqrt(sigma2_mean_sim / sigma2_signal)

    return scale_factor

#-------------------------------------------------------------------------------
# Long integrations light curve
#-------------------------------------------------------------------------------
def simulateLightCurveSamplingAverage(t, dt,
                                      tSamp=0.1,
                                      beta=1.0,
                                      real_imag_ratio=1.0e9,
                                      tSim=365.0 * 20.0,
                                      index_sampling=None):
    """ Simulate light curve with a given PSD exponent and sample as the input 
    light curve averaging several points to fill t +/- dt intervals

    The output times are taken from the data but the samples are generated at 
    slightly different times. The difference is smaller than the samping time 
    so it might produce spourious high frequency components. I expect the effect
    not to be very large.

    input
    -----
    t                      Data set to get sampling from
    dt                     Time interval for average
    tSamp=1.0              Sampling time (I think as units in days)
    beta=1.0               Exponent of PSD (1/f^beta)
    real_imag_ratio        Upper limit for imaginary part on time series
                           See oofNoise() for details
    tSim                   Time lenght of simulates light curve. It has to be 
                           al least as long as t
    index_sampling=array([])        
                           Sampling pattern for light curve. It can be used when
                           using function to simulate many light curves
                           I use None to avoid modifying mutable object if I 
                           don't define default parameter on call

    output
    ------
    t               Time vector from actual data
    s_s             Simulated data with given PSD and sampled as data
    t_sim, s_sim    Finely sampled simulated light curve
    index_sampling=None  index sampling used. It speeds computation up. It has
                         to be initialized as array([]) and returns used one.

    notes
    -----
    - index_sampling is a list of arrays with each element contaning an array
    with indices for the elements to average on  respective sample


    Walter Max-Moerbeck, September 27, 2012
    """

    # number of samples required for one light curve
    # Find maximum among lenght of t and tSim
    # This time lenght considers the time added by interval width at extremes
    time_length_t = (t[-1] + dt[-1]) - (t[0] - dt[0])

    if time_length_t >= tSim:

        time_length = time_length_t

    else:
        
        time_length = tSim    


    # Calculate minimum number of samples needed from simulation    
    N = int(ceil(( time_length / tSamp + 1)))

    # make it a power of two to improve speed of fft
    N = 2 ** int(ceil(log2(N)))

    # simulate finely sampled light curve with same time duration
    t_sim, s_sim = oofNoise(N,
                            tSamp=tSamp,
                            beta=beta,
                            real_imag_ratio=real_imag_ratio)

    # Take a section of the light curve to sample
    N_original =  int(ceil(time_length_t / tSamp + 1))
    t_sim = t_sim[:N_original]
    s_sim = s_sim[:N_original]

    # sample light curve as the data
    # make first simulated sample time equal to data time at extreme of first 
    # interval
    t_sim = t_sim - t_sim[0] + min(t) - dt[0]
 

    # Get simulated elements by one
    t_s = t[:]

    # Do case with known index_sampling
    if len(index_sampling) == len(t):

        # Take mean value of elements that make this integration
        s_s = scipy.array([scipy.mean(s_sim[index_sampling_i]) for
                           index_sampling_i in index_sampling])
        
    # Or find the index_sampling patterns for each data point
    else:

        # Create array to save sampling patterns
        index_sampling = []

        # Construct sampling pattern
        for i in xrange(len(t_s)):

            # Find elements that into average for point i
            index_sampling_i = scipy.where((t[i] - dt[i] <= t_sim) &
                                           (t_sim < t[i] + dt[i]))[0]

            assert (len(index_sampling_i) > 0),\
                'No samples to average. tSamp is too long. Try a shorter tSamp.'

            index_sampling.append(index_sampling_i)

        # Take mean value of elements that make this integration
        s_s = scipy.array([scipy.mean(s_sim[index_sampling_i]) for
                           index_sampling_i in index_sampling])

    return t_s, s_s, t_sim, s_sim, index_sampling


#-------------------------------------------------------------------------------
def getErrorScaleFactorAverage(t, s, error, dt,
                               tSamp=1.0,
                               beta=1.0,
                               tSim=0.0,
                               N_sim=100,
                               real_imag_ratio=1.e9):
    """ Simulate N_sim light curves with given PSD to get approximate
    normalization for the case of long integrations

    If there is more error than signal set scale_factor = 0.0, in which case 
    only the noise is modeled

    This function is an exact copy of getErrorScaleFactor() but for the case
    of long integrations so it uses simulateLightCurveSamplingAverage() instead
    of simulateLightCurveSampling()


    input
    -----
    t, s, error, dt        Time series with error. dt is HW of time bins 
    tSamp=1.0              Sampling period for simulated light curves
    beta=1.0               Power-law index of simulated light curves
    tSim=0.0               Lenght of simulated light curves
    N_sim=100              Number of simulated light curves
    real_imag_ratio=1.e9   Ensure small imaginary part in simulated time series

    output
    ------
    scale_factor           Factor by which the error on data has to be multiplied
                           to get same relative contribution to simulated data

    notes
    -----
    - The idea is that the variance in the data has two independent components,
    the signal and the noise. If this is the case
    
    var(data) = var(signal) + var(noise)

    This function finds the conversion factor between simulated variance, which
    is assumed to be pure signal and var(signal)

    scale_factor = sqrt(var(sim) / var(signal))



    Walter Max-Moerbeck, February 14, 2011
    """

    # Signal variance
    sigma2_data = var(s, ddof=1)
    sigma2_noise = mean(error**2)

    sigma2_signal = sigma2_data - sigma2_noise

    # Check the anomalous case when sigma2_signal <= 0.0 and set scale to 0.0
    if sigma2_signal <= 0.0:
        
        scale_factor = 0.0

        print 'sigma2_data = %f, sigma2_signal = %f, sigma2_noise = %f' %\
            (sigma2_data,
             sigma2_signal,
             sigma2_noise)

        sys.exit("getErrorScaleFactorAverage() error. sigma2_signal <= 0.0")

        return scale_factor


    # Simulate light curve and get sigma2 for each one
    SIGMA2_SIM = []

    # Index sampling array to speed up computation
    index_sampling = []

    for i in xrange(N_sim):

        # Simulate light curve with given sampling
        t_s, s_s, t_sim, s_sim, index_sampling =\
            simulateLightCurveSamplingAverage(t, dt,
                                              tSamp=tSamp,
                                              beta=beta,
                                              real_imag_ratio=real_imag_ratio,
                                              tSim=tSim,
                                              index_sampling=index_sampling)
        
        # Get sigma^2 for simulation with given sampling
        SIGMA2_SIM.append(var(s_s))

    # Compute scale factor from the mean sigma2 for simulations and data
    sigma2_mean_sim = mean(SIGMA2_SIM)

    # Get scale factor for error bars on simulated data
    # error_s_s = scale_factor * error_s
    scale_factor = sqrt(sigma2_mean_sim / sigma2_signal)

    return scale_factor


#-------------------------------------------------------------------------------
# Data randomization
#-------------------------------------------------------------------------------
def random_subset(t, f, e):
    """ Selects a random subsample of the data.
    
    Points are selected randomly with repetition. Repeated selections are
    used to reduce errors in data points
    

    INPUT
    -----
    t, f, e      Time, flux and flux error for time series


    OUTPUT
    ------
    t, f, e      Subset of the original time series


    NOTES
    -----

    Random subset selection and Flux ramdomization light curves
    Uses ideas from Peterson et al 1998. PASP, 110, 660

    This version modifies error on points that are selected multiple times as
    suggested by Welsh 1999, PASP, 111, 1347

    The modifications are confirmed to be good by Peterson et al 2004, ApJ
    613, 682

    
    Walter Max-Moerbeck, July 9, 2009.
    """

    #----------------------------------------
    # Choose elements with repetitions and eliminates repetitions
    N = len(t)
    index = random.randint(0, N, N)
    index_unique = unique(index)

    #----------------------------------------
    # Select the appropriate data set
    t_s = t[index_unique]
    f_s = f[index_unique]

    e_s =\
        array([e[i] / sqrt(len(where(i == index)[0])) for i in\
               index_unique])
    

    return t_s, f_s, e_s


#-------------------------------------------------------------------------------
def random_subset_eliminate(t, f, e):
    """ Selects a random subsample of the data.
    
    Points are selected randomly with repetition. Repeated selections are
    elimnated
    

    INPUT
    -----
    t, f, e      Time, flux and flux error for time series


    OUTPUT
    ------
    t, f, e      Subset of the original time series


    NOTES
    -----

    Random subset selection and Flux ramdomization light curves
    Uses ideas from Peterson et al 1998. PASP, 110, 660

    
    Walter Max-Moerbeck, July 9, 2009.
    """

    #----------------------------------------
    # Choose elements with repetitions and eliminates repetitions
    N = len(t)
    index = random.randint(0, N, N)
    index_unique = unique(index)
    
    #----------------------------------------
    # Select the appropriate data set and weight errors for repetitions
    t_s = t[index_unique]
    f_s = f[index_unique]
    e_s = e[index_unique]
    

    return t_s, f_s, e_s


#-------------------------------------------------------------------------------
def randomize_flux(t, f, e):
    """ Randomize the flux measurements adding noise according to the error bars
    Use a normal distribution of flux errors with sigma give by the error bar

    It can handle values e >= 0


    INPUT
    -----
    t, f, e      Time, flux and flux error for time series
    
    OUTPUT
    ------
    t, f, e      Time, flux and flux error for time series with randomized
                 fluxes

    NOTES
    -----

    Random subset selection and Flux ramdomization light curves
    Uses ideas from Peterson et al 1998. PASP, 110, 660  


    Walter Max-Moerbeck, July 9, 2009.
    """

    #----------------------------------------
    # generate random error with appropriate distribution and add to data points
    f_r = f + e * random.normal(loc=0.0, scale=1.0, size=size(f))

    return t, f_r, e 


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------
def oofNoise_test():
    """ Test that the noise generated by the function has the claimed Fourier 
    transform and the right sampling

    Walter Max-Moerbeck, April 21, 2010
    """
    
    # number of samples on simulated light curve
    N = 10000

    # test a few values for the exponent
    for beta in [0.0, 1.0, 2.0, 4.0]:
        for tSamp in [0.1, 1.0, 10.0]:

            # test signal and Fourier Transform
            t, f = oofNoise(N, tSamp=tSamp, beta=beta)

            ft = fftpack.fft(f)
            ft_freq = arange(N/2 + 1) / (N * tSamp)
            ft_freq = concatenate((ft_freq, ft_freq[N/2 - 1:0:-1]))

            # Plots
            pylab.clf()
            polycoeffs = scipy.polyfit(scipy.log10(ft_freq[1:N/2]),
                                       scipy.log10(abs(ft[1:N/2])), 1)

            ffit = scipy.polyval(polycoeffs, scipy.log10(ft_freq[1:N/2]))
        
            pylab.plot(scipy.log10(ft_freq[1:N/2]),\
                           scipy.log10(abs(ft[1:N/2])))
            pylab.plot(scipy.log10(ft_freq[1:N/2]), ffit)

            pylab.title('initial: ' + str(-beta) +\
                            ' fitted: ' + str(2 * polycoeffs[0]))

            fig_title = 'crossCorr_tests/oofNoise_test/noise_%s_%s.png' %\
                (str(beta), str(tSamp))
            pylab.savefig(fig_title)

    return


#-------------------------------------------------------------------------------
def simulateLightCurveSampling_test(test_results_folder='test/'):
    """ Plot a few sample cases to check function is working properly

    Walter Max-Moerbeck, April 29, 2010
    """

    # generate a test signal
    t = scipy.rand(100)
    t.sort()
    s = scipy.sin(t * 2 * scipy.pi)

    # noise light curves with same sampling
    for i in xrange(100):
        
        t_s, s_s, t_sim, s_sim, index_sampling =\
            simulateLightCurveSampling(t,
                                       tSamp=0.01,
                                       beta=2.0,
                                       real_imag_ratio=1.0e9,
                                       tSim=100,
                                       index_sampling=array([]))
        
        pylab.clf()
        pylab.subplot(211)
        pylab.plot(t, s, '.-', label='data')

        # scale for better visualisation
        sfact = max(s) / max(s_sim)

        pylab.plot(t_sim, s_sim * sfact, '.-',label='sim fine sampling')
        pylab.plot(t_s, s_s * sfact, '.-',label='sim and sampled')
        pylab.legend(loc=0, frameon=False)
        pylab.xlabel('t')
        pylab.ylabel('s')

        pylab.subplot(212)
        pylab.plot(t, t_s, '.-')
        pylab.xlabel('t')
        pylab.ylabel('t_s')
        pylab.savefig(test_results_folder + 'slcs_%s.png' % (str(i)))

    return


#-------------------------------------------------------------------------------
def simulateLightCurveSamplingAverage_test(test_results_folder='test_average/'):
    """ Plot a few sample cases to check function is working properly

    Walter Max-Moerbeck, September 13, 2010
    """

    # generate a test signal
    t = scipy.arange(0, 101, 10)
    dt = scipy.ones(len(t)) * 5.0
    s = scipy.sin(t * 2 * scipy.pi)

    # noise light curves with same sampling
    for i in xrange(10):
        
        t_s, s_s, t_sim, s_sim, index_sampling =\
            simulateLightCurveSamplingAverage(t, dt,
                                              tSamp=1,
                                              beta=2.0,
                                              real_imag_ratio=1.0e9,
                                              tSim=365.0 * 20.0,
                                              index_sampling=[])
            
        t_s, s_s, t_sim, s_sim, index_sampling =\
            simulateLightCurveSamplingAverage(t, dt,
                                              tSamp=1,
                                              beta=2.0,
                                              real_imag_ratio=1.0e9,
                                              tSim=365.0 * 20.0,
                                              index_sampling=index_sampling)
            
        pylab.clf()
        #pylab.plot(t, s, '.-', label='data')

        # scale for better visualisation
        sfact = max(s) / max(s_sim)

        pylab.plot(t_sim, s_sim * sfact, '.-',label='sim fine sampling')
        pylab.plot(t_s, s_s * sfact, '.-',label='sim and sampled')
        pylab.legend(loc=0, frameon=False)
        pylab.xlabel('t')
        pylab.ylabel('s')
        pylab.savefig(test_results_folder + 'slcsa_%s.png' % (str(i)))

    return
