"""
OVRO 40m program variability
 
Routines for correlation and cross-correlation of unevenly spaced data.
 
This verson is the same as crossCorrLCCF but it aasumes that one the time series,
the b-time-series is of the long integration type.

Walter Max-Moerbeck, 13 May 2009
"""


from scipy import array, array_equal, mean, std, arange, random, ones, sqrt,\
    zeros, ceil, concatenate, where, absolute, unique, stats, concatenate,\
    argsort, argmin, median, floor, sum, var, special, transpose, nan, isnan

import pylab

from tsTools.utils import bin_data, detrend_linear, bin_data, savefile, savefig

from tsTools import simData


import os

import pickle


#-------------------------------------------------------------------------------
# Basic computation
#-------------------------------------------------------------------------------
def crosscorr(ta, fa, erra,
              tb, fb, errb,
              delta_t,
              error_corr=False,
              min_elem_bin=2,
              linear_detrending=False):
    """ Local cross correlate two unevenly sampled time series

    The idea is to have a plug-and-play replacement for the cross-corr but that
    implements the local CCF (Welsh 1999). Such replacements will be needed
    to implement upper-limits and other modifications

    Input
    -----
    ta, fa, erra:  Data for first time series

    tb, fb, errb:  Data for second time series

    delta_t:       Width of time bins

    error_corr:    True: Apply right normalization and zero lag correction
                         See note below
                   False: Ignore these as most people do

    min_elem_bin:  Minimum number of elements per DCF bin 

    linear_detrend: Detrend time series with a linear fit


    Output
    ------
    TAU, SIGMA_TAU, DCF, SIGMA_DCF, N_ELEM   
                   Cross correlation at different time lags

    Notes
    -----
    - Make selection of overlaping samples faster

    Walter Max-Moerbeck, Dec 29, 2011.
    """

    #--------------------------------------------------
    # Detrend with linear fit if requested
    if linear_detrending == True:
        fa = detrend_linear(ta, fa)
        fb = detrend_linear(tb, fb)


    #--------------------------------------------------
    # Find all posible time lags and associated data pairs
    DELTA_t = array([tb_j - ta_i\
                     for ta_i in ta\
                     for tb_j in tb])
    
    PAIRS_ab = array([[a_i, b_j]\
                      for a_i in fa\
                      for b_j in fb])

    delta_t_2 = 0.5 * delta_t

    
    #--------------------------------------------------
    # Make a list of time bins centers that always has a bin centered in 0
    # find maximum bin index
    tmax = max(DELTA_t)
    tmin = min(DELTA_t)
    
    N = int(ceil(tmax / delta_t - 0.5))
    
    # find minimum bin index
    M = int(floor(tmin / delta_t + 0.5))
    
    # array of bin centers
    bin_center = arange(M, N + 1.0) * delta_t

    
    #----------------------------------------------
    # For each bin find cross-correlation

    # Arrays to save results, initialized for speed
    N_bins = len(bin_center)
    TAU = zeros(N_bins)
    SIGMA_TAU = zeros(N_bins)
    DCF = zeros(N_bins)
    SIGMA_DCF = zeros(N_bins)
    N_ELEM = zeros(N_bins)

    # Count number of bins with minimum number of elements
    i_bin = 0

    for bin_tau in bin_center:

        # Find elements with that delay
        index_tau = where((bin_tau - delta_t_2 <= DELTA_t) &
                          (DELTA_t < bin_tau + delta_t_2))[0]

        # Check it has minimum number of elements
        n_elem = len(index_tau)
        if n_elem < min_elem_bin:
            continue

        pairs_tau = PAIRS_ab[index_tau]

        # Put them into separate arrays
        x = array([pair[0] for pair in pairs_tau])
        y = array([pair[1] for pair in pairs_tau])


        # Correlate them
        r = sum((x - mean(x)) * (y - mean(y))) / n_elem /\
            sqrt(var(x, ddof=0) * var(y, ddof=0))
        
        
        # Save results to array and update counter
        TAU[i_bin] = bin_tau
        SIGMA_TAU[i_bin] = delta_t
        DCF[i_bin] = r
        SIGMA_DCF[i_bin] = 0
        N_ELEM[i_bin] = n_elem

        i_bin += 1
    
    # Truncate to eliminate non-used elements
    TAU = TAU[:i_bin]
    SIGMA_TAU = SIGMA_TAU[:i_bin]
    DCF = DCF[:i_bin]
    SIGMA_DCF = SIGMA_DCF[:i_bin]
    N_ELEM = N_ELEM[:i_bin]


    return TAU, SIGMA_TAU, DCF, SIGMA_DCF, N_ELEM
    

#-------------------------------------------------------------------------------
# Significance evaluation
#-------------------------------------------------------------------------------
def crosscorr_significance_plaw_noises(ta, fa, ea,
                                       tb, fb, eb, dtb,
                                       beta_a=1.0,
                                       tSamp_a=1.0,
                                       tSim_a=10.0,
                                       beta_b=1.0,
                                       tSamp_b=1.0,
                                       tSim_b=10.0,
                                       delta_t=10.0,
                                       N_pairs=1000,
                                       mock_test=False,
                                       Dt_proj_a=365. * 6,
                                       Dt_proj_b=365. * 5,
                                       min_elem_bin_xcorr=2,
                                       linear_detrending_xcorr=False,
                                       error_corr_xcorr=False,
                                       N_sim_scale_factor=100,
                                       add_obs_noise=False):
    """ Evaluate significance using randomly generated samples with the same 
    sampling as the actual light curves. 
    
    Uses power law noise for the mock light curves.

    This version is for one of the light curves being a long integration as are
    gamma-ray light curves

    input
    -----
    ta, fa, ea          Light curves for cross-correlation
    tb, fb, eb, dtb     They are refered as 'a' and 'b'

    beta_a         Power law for noise 1/f^beta for ligth curve 'a'
    beta_b         Same for light curve 'b'

    tSamp_a        Sampling time for simulated data. Units are arbitrary
    tSamp_b        

    tSim_a         Time lenght of simulated light curve
    tSim_b

    delta_t        Bin size in days for cross-correlations

    N_pairs        Mock data sets used for significance evaluation

    mock_test      When True, plots the simulated mock data sets, calculate 
                   the power spectrum of each one save results to a file.
                   This is only meant to be used during testing.
                   - See mock_test_analyze() on this module for a fucntion to 
                   analyze the results

    D_proj_a       Light curve time base in days for significance projections

    D_proj_b       Light curve time base in days for significance projections

    min_elem_bin_xcorr=2            Minimum number of elements for time lag bin

    linear_detrending_xcorr=False   Linear detrending for cross-correlation

    error_corr_xcorr=False          Error correction on cross-correlation
                                    normalization

    N_sim_scale_factor=100          Number of simulations for observing noise
                                    scale factor

    add_obs_noise=False             Add observational noise to simulations


    output
    ------
    results        Dictionary with results of the significance analysis and 
                   cross-correlation.


    notes
    -----
    - Since the errors are not used it uses the simple version of the
    cross-correlation, crosscorr()

    - Observational errors are not included for projections

    - b light curve is of the long integration type. The time bins are assumed 
    to be of the same lenght. The fucntion has to be modified to include more
    generality
    

    Walter Max-Moerbeck, January 8, 2010
    """ 

    #----------------------------------------
    # get cross-correlation with errors for the data
    tau_0, sigma_tau_0, dcf_0, sigma_dcf_0, n_elem_0 =\
           crosscorr(ta, fa, ea,
                     tb, fb, eb,
                     delta_t=delta_t,
                     error_corr=error_corr_xcorr,
                     min_elem_bin=min_elem_bin_xcorr,
                     linear_detrending=linear_detrending_xcorr)
    
    
    #----------------------------------------
    # Increase light curve lenght for significance projections if requested
    ta_ext = enlarge_time_vector(ta, Dt_proj_a)

    # Assumes all the time bins are equal
    tb_ext = enlarge_time_vector(tb, Dt_proj_b)
    dtb_ext = ones(len(tb_ext)) * dtb[0]


    #----------------------------------------
    # Calculate scaling factors for observational error
    if add_obs_noise == True:
        scale_factor_a = simData.getErrorScaleFactor(ta, fa, ea,
                                                     tSamp=tSamp_a,
                                                     beta=beta_a,
                                                     tSim=tSim_a,
                                                     N_sim=N_sim_scale_factor,
                                                     real_imag_ratio=1.e9)
        
        scale_factor_b =\
            simData.getErrorScaleFactorAverage(tb, fb, eb, dtb,
                                               tSamp=tSamp_b,
                                               beta=beta_b,
                                               tSim=tSim_b,
                                               N_sim=N_sim_scale_factor,
                                               real_imag_ratio=1.e9)
        
    else:
        scale_factor_a = 0.0
        scale_factor_b = 0.0


    #----------------------------------------
    # evaluate significance

    # arrays to store results
    TAU = array([])
    DCF = array([])

    # Sampling patterns, initially unknown
    # b light curve is of long integration type, use [] instead of array([])
    index_sampling_a = array([])
    index_sampling_b = []
    

    # generate mock data sets and correlate
    for i in xrange(N_pairs):
        
        # generate power law noise samples
        
        #--------------------
        # Light curve 'a'
        ta_s, fa_s, t_sim, s_sim, index_sampling_a =\
              simData.simulateLightCurveSampling(ta_ext,
                                                 tSamp=tSamp_a,
                                                 beta=beta_a,
                                                 tSim=tSim_a,
                                                index_sampling=index_sampling_a)

        # Add observational noise
        fa_s = simData.addObservingNoise(ta, fa, ea,
                                         ta_s, fa_s,
                                         scale_factor=scale_factor_a)
        
        
        
        #--------------------
        # Light curve 'b'
        tb_s, fb_s, t_sim, s_sim, index_sampling_b =\
              simData.simulateLightCurveSamplingAverage(tb_ext, dtb_ext,
                                                        tSamp=tSamp_b,
                                                        beta=beta_b,
                                                        tSim=tSim_b,
                                                index_sampling=index_sampling_b)

        # Add observational noise
        fb_s = simData.addObservingNoise(tb, fb, eb,
                                         tb_s, fb_s,
                                         scale_factor=scale_factor_b)
        
        
        #--------------------
        # correlate
        # Errors are not used in crosscorr() bogus values are enter
        tau, sigma_tau, dcf, sigma_dcf, n_elem =\
             crosscorr(ta_s, fa_s, 0.05 * absolute(fa_s),
                       tb_s, fb_s, 0.05 * absolute(fb_s),
                       delta_t=delta_t,
                       error_corr=False,
                       min_elem_bin=min_elem_bin_xcorr,
                       linear_detrending=linear_detrending_xcorr)
        
        # add to array
        TAU = concatenate((TAU, tau[:]))
        DCF = concatenate((DCF, dcf[:]))
        
        #--------------------
        # save mock data if requested
        #if mock_test == True:
        #    try:
        #        mock_results.append({'ta_s': ta_s,
        #                             'fa_s': fa_s,
        #                             'tb_s': tb_s,
        #                             'fb_s': fb_s,
        #                             'tau': tau,
        #                             'sigma_tau': sigma_tau,
        #                             'dcf': dcf,
        #                             'sigma_dcf': sigma_dcf
        #                             })
        #    except NameError:
        #        mock_results = []
        #        mock_results.append({'ta_s': ta_s,
        #                             'fa_s': fa_s,
        #                             'tb_s': tb_s,
        #                             'fb_s': fb_s,
        #                             'tau': tau,
        #                             'sigma_tau': sigma_tau,
        #                             'dcf': dcf,
        #                             'sigma_dcf': sigma_dcf
        #                             })
                
    #----------------------------------------
    # calculate significance levels
    tau_bins, sigma_levels, mean_level, perc_dcf_0 =\
              confidence_intervals(TAU, DCF, tau_0, dcf_0)


    #--------------------------------------------------
    # make a dictionary with data to return which define simulation
    notes = """ Description of the data saved on this dictionary

    Cross-correlation results
               'tau_0': tau_0,
               'sigma_tau_0': sigma_tau_0,
               'dcf_0': dcf_0,
               'sigma_dcf_0': sigma_dcf_0,
               'n_elem_0': n_elem_0,
                  
    Significance results, reference levels and percentiles for data cross-corre
lation
               'tau_bins': tau_bins,
               'sigma_levels': sigma_levels,
               'perc_dcf_0': perc_dcf_0, 

    Simulated cross-correlations
               'TAU': TAU,
               'DCF': DCF,
    """
    

    results = {'tau': tau_0,
               'sigma_tau': sigma_tau_0,
               'dcf': dcf_0,
               'sigma_dcf': sigma_dcf_0,
               'dcf_nelem': n_elem_0,
               'tau_bins': tau_bins,
               'sigma_levels': sigma_levels,
               'mean_level': mean_level,
               'perc_dcf': perc_dcf_0, 
               'TAU': TAU,
               'DCF': DCF}
    

    #--------------------
    # return mock test results if requested
    #if mock_test == True:
    #    results['mock_test'] = mock_results

    return results


#-------------------------------------------------------------------------------
def crosscorr_significance_plaw_noises_plot(tau, dcf, sigma_dcf,
                                            tau_bins, sigma_levels, mean_level,
                                            perc_dcf,
                                            plot_title='',
                                            plots_folder='',
                                            file_extensions=['.png',
                                                             '.pdf',
                                                             '.eps'],
                                            y_lim=(-1.2, 1.2),
                                            x_lim=None,
                                            save_data=True):
    """ Summary plots for cross-correlation significance analysis

    notes
    -----
    - I could add a text note to the figure with information on the best peak
    
    best_peak_info = {'tau': tau,
                      'tau_std': tau_std,
                      'dcf_sig': dcf_sig}

    Walter Max-Moerbeck, June 15, 2011
    """
    
    #----------------------------------------
    # plot cross-correlation and significance levels
    pylab.clf()
    pylab.errorbar(tau, dcf, sigma_dcf,
                   fmt='.k',
                   label='cross-corr',
                   capsize=0)

    for sigma_lev in sigma_levels:
        pylab.plot(tau_bins, sigma_lev['sigma_pos'],
                   sigma_lev['line_type'],
                   color=sigma_lev['line_color'],
                   label=str(sigma_lev['level']),
                   lw=2)

        pylab.plot(tau_bins, sigma_lev['sigma_neg'],
                   sigma_lev['line_type'],
                   color=sigma_lev['line_color'],
                   label=str(sigma_lev['level']),
                   lw=2)

    # plot mean value to check distribution is symmetric
    pylab.plot(tau_bins, mean_level, '-.k', label='mean')

    # title and legends
    #pylab.legend(loc=0)
    pylab.title(plot_title)
    pylab.xlabel(r'$\tau$ [days]')
    pylab.ylabel('CCF')

    # set y-scale
    pylab.ylim(y_lim)

    # set x-scale, x_lim=None by default shows all data
    if x_lim == None:
        pylab.xlim((min(tau), max(tau)))

    else:
        pylab.xlim(x_lim)


    #pylab.grid()

    # Save plots in multiple formats
    plot_xcorrsig_out = os.path.join(plots_folder,
                                 plot_title.replace(' ', '_').replace('.', 'p'))

    for extension in file_extensions:
        pylab.savefig(plot_xcorrsig_out + extension)


    # Save the data on text files if requested
    if save_data == True:
        savefile(plot_xcorrsig_out + '_xcorr.txt',
                 data=transpose((tau, dcf, sigma_dcf)),
                 fmt='%f',
                 header='#tau, dcf, sigma_dcf\n')

        for sigma_lev in sigma_levels:
            savefile(plot_xcorrsig_out + '_dist_%f.txt' % (sigma_lev['level']),
                     data=transpose((tau_bins,
                           sigma_lev['sigma_pos'],
                           sigma_lev['sigma_neg'])),
                     fmt='%f',
                     header='#tau, dcf_neg, dcf_pos\n')

        savefile(plot_xcorrsig_out + '_dist_mean.txt',
                 data=transpose((tau_bins,
                       mean_level)),
                 fmt='%f',
                 header='#tau, dcf\n')



    #----------------------------------------
    # plot cross-correlation and percentiles
    pylab.clf()
    pylab.plot(tau, dcf, '.-b', label='xcorr')

    pylab.plot(array(perc_dcf['tau']), array(perc_dcf['perc']) / 100.0,
               '-k', label='percentiles')


    # title and legends
    pylab.legend(loc=0)
    pylab.title(plot_title)
    pylab.xlabel(r'$\tau$ [days]')
    pylab.ylabel('CCF')

    # set y-scale
    pylab.ylim(y_lim)

    # set x-scale, x_lim=None by default shows all data
    if x_lim == None:
        pylab.xlim((min(tau), max(tau)))

    else:
        pylab.xlim(x_lim)


    pylab.grid()

    # Save plots in multiple formats
    plot_xcorrsig_perc_out = os.path.join(plots_folder,
                               plot_title.replace(' ', '_').replace('.', 'p') +\
                                          '_percentiles')

    for extension in file_extensions:
        pylab.savefig(plot_xcorrsig_perc_out + extension)


    #----------------------------------------
    # plot pvalues
    # Get sigma limits
    sigma = arange(1.0, 4.0, 1.0)
    p_sigma = special.erfc(sigma / sqrt(2.0))

    pylab.clf()
    pylab.plot(array(perc_dcf['tau']),
               1.0 - array(perc_dcf['perc']) / 100.0,
               '-k')


    # set x-scale, x_lim=None by default shows all data
    if x_lim == None:
        pylab.xlim((min(tau), max(tau)))

    else:
        pylab.xlim(x_lim)

    # get limits to put text on right y-axis
    x_min, x_max = pylab.xlim()


    for p_s, sigma_s in zip(p_sigma, sigma):
        pylab.axhline(p_s, ls='--', color='k')
        pylab.text(x_max, p_s, r'%d$\sigma$' % (sigma_s))


    # title and legends
    pylab.title(plot_title)
    pylab.xlabel(r'$\tau$ [days]')
    pylab.ylabel('p')

    # set y-scale
    pylab.yscale('log')

    # Save plots in multiple formats
    plot_xcorrsig_perc_out = os.path.join(plots_folder,
                               plot_title.replace(' ', '_').replace('.', 'p') +\
                                          '_pvalues')

    for extension in file_extensions:
        pylab.savefig(plot_xcorrsig_perc_out + extension)    


    # Save percentiles to a text file
    header = '# Percentiles for cross-correlation\n'
    header = '#tau,p\n'

    file_name = '%s.txt' % (plot_xcorrsig_perc_out)
    savefile(file_name,
             data=transpose((perc_dcf['tau'],
                             1.0 - array(perc_dcf['perc']) / 100.0)),
             fmt='%f',
             header=header)

    return


#-------------------------------------------------------------------------------
def enlarge_time_vector(t, time_span_new):
    """ Given a time vector and time base, it enlarges or shrinks time series 
    using repetitions of the same sampling pattern

    input
    -----
    t                Sampling time series
    time_span_new    Requested time span

    output
    ------

    notes
    -----
    - This function was written for use with the correlations significance 
    projections

    Walter Max-Moerbeck, May 12, 2010
    """
    
    # Convert to scipy array
    t = array(t)

    if time_span_new == 0:
        t_new = t[:]

    else:
        # get time span for original sampling times
        time_span_orig = max(t) - min(t)
    
        # get ceil for increase factor
        N_proj = int(ceil(time_span_new / time_span_orig))

        # concatenate the pieces
        t_new = t[:]
        for i in range(N_proj - 1):
            t_new = concatenate((t_new, t[1:] - t[0] + t_new[-1]))

        # cut the sampling time series to requested length
        t_new = t_new[where((t_new - t_new[0]) <= time_span_new)[0]]

    return t_new


#-------------------------------------------------------------------------------
def confidence_intervals(TAU, DCF, tau, dcf,
                         levels=[68.27, 95.45, 99.73],
                         line_type=[':', '-.', '--'],
                         line_color=['#9C0000', '#FF6F00', '#006300']):
    """ Calculate confidence intervals for significance analysis of correlations

    input
    -----
    TAU, DCF     Cross-correlations and time lags for mock data sets
    
    output
    ------

    tau_bins                    Time lags on the input data
    sigma_1, sigma_2, sigma_3   Confidence intervals 
    

    Walter Max-Moerbeck, January 13, 2010
    """

    # array to save results
    sigma_levels = []

    # Calculate confidence limits for each different time lag bin
    
    tau_bins = unique(TAU)

    for level, line, color in zip(levels, line_type, line_color):

        sigma_pos = []
        sigma_neg = []
        for tau_bin in tau_bins:
            index_pos = where((TAU == tau_bin) & (DCF >= 0))[0]
            index_neg = where((TAU == tau_bin) & (DCF < 0))[0]

            # calculate 1-sigma, 2-sigma and 3-sigma levels
            s_pos = stats.scoreatpercentile(DCF[index_pos], level)
            s_neg = stats.scoreatpercentile(-DCF[index_neg], level) 

            sigma_pos.append(s_pos)
            sigma_neg.append(-s_neg)

        sigma_level = {'level': level,
                       'line_type': line,
                       'line_color': color,
                       'sigma_pos': sigma_pos,
                       'sigma_neg': sigma_neg}

        sigma_levels.append(sigma_level)


    # Calculate percentile of data cross-correlation wrt simulations 
    # A problem is found when dcf_bin is larger than all the simulations, in 
    # this case we can't conclude that perc = 100.0, but instead we are limited
    # by the number of simulations to say perc = 100.0 - 100. / N
    perc_tau = []
    perc_dcf = []

    for tau_bin, dcf_bin in zip(tau, dcf):

        index_pos = where((TAU == tau_bin) & (DCF >= 0))[0]
        index_neg = where((TAU == tau_bin) & (DCF < 0))[0]

        # Check there are elements or skip
        if len(index_pos) == 0  and len(index_neg) == 0:
            continue
        
        if dcf_bin >= 0:

            perc =\
                100. * len(where(DCF[index_pos] < dcf_bin)[0]) / len(index_pos)

            # correct for finite number of simulations
            if perc == 100.0:
                perc = 100.0 - 100.0 / len(index_pos)
                

        else:
            perc =\
               100. * len(where(-DCF[index_neg] < -dcf_bin)[0]) / len(index_neg)
            
            # correct for finite number of simulations
            if perc == 100.0:
                perc = 100.0 - 100.0 / len(index_neg)


        # add to array
        perc_tau.append(tau_bin)
        perc_dcf.append(perc)
        

    perc_dcf = {'tau': perc_tau,
                'perc': perc_dcf}

    # Calculate mean value to check distribution is symmetric around zero
    mean_level = []
    for tau_bin in tau_bins:
        index = where(TAU == tau_bin)[0]

        mean_in_bin = mean(DCF[index])
        mean_level.append(mean_in_bin)


    return tau_bins, sigma_levels, mean_level, perc_dcf


#-------------------------------------------------------------------------------
# Analyze results
#-------------------------------------------------------------------------------
def error_on_peak_best(ta, fa, erra,
                       tb, fb, errb,
                       tau, dcf, sigma_dcf, dcf_nelem,
                       TAU, DCF,
                       delta_t,
                       min_elem_bin,
                       linear_detrending,
                       N_pairs,
                       N_peaks=3,
                       N_sim=1000,
                       N_bootstrap_sig=1000,
                       plots_folder='',
                       plot_title='',
                       file_extensions=['.png',
                                        '.pdf',
                                        '.eps'],
                       lags_range=(0.0, 0.0)):
    """ Using RSS and FR find the error in the peak for all the
    cross-correlation peaks

    input
    -----
    ta, fa, erra     Time series data
    tb, fb, errb

    tau, dcf, sigma_dcf, dcf_nelem    Cross-correlation results 

    TAU, DCF    Cross-correlation significance results

    delta_t

    N_peaks

    N_sim

    plots_folder

    plot_title

    file_extensions

    lags_range       Range of time lags to search for cross-correlation peaks


    output
    ------
    results_peaks_best


    notes
    -----

    

    Walter Max-Moerbeck, May 25, 2011
    """


    # Find all peaks on cross-correlaton
    results_peaks = find_peaks_on_xcorr(tau, dcf, sigma_dcf, dcf_nelem,
                                        TAU, DCF,
                                        lags_range=lags_range)
    

    # updated version with results for each peak
    results_peaks_best = []


    # Find error for each peak and on significance
    i = 0
    for r in results_peaks:

        # do only first N_peaks peaks
        if i >= N_peaks:
            break
        else:
            i += 1

        # Time lag, dcf and significance for peak
        tau_peak = r['tau']
        dcf_peak = r['dcf']
        dcf_sig = r['dcf_sig']

        #----------------------------------------
        # Error on time lag
        # Plot name for this particular peak
        plot_title_tau = '%s %f' % (plot_title, tau_peak)

        tau_std, tau_peak_median, tau_std_ll, tau_std_ul, TAU_PEAK =\
                 error_on_peak(ta, fa, erra,
                               tb, fb, errb,
                               delta_t,
                               min_elem_bin,
                               linear_detrending,
                               tau_peak,
                               N_sim=N_sim)
        
        error_on_peak_plot(tau, dcf, sigma_dcf,
           tau_peak, tau_std, tau_peak_median, tau_std_ll, tau_std_ul, TAU_PEAK,
                           plot_title=plot_title_tau,
                           plots_folder=plots_folder,
                           file_extensions=file_extensions,
                           lags_range=lags_range)

        
        # Add elements to dictionary
        r['tau_std'] = tau_std
        r['tau_peak_median'] = tau_peak_median
        r['tau_std_ll'] = tau_std_ll
        r['tau_std_ul'] = tau_std_ul


        #----------------------------------------
        # Error on significance
        sig_median, sig_error, P =\
                    bootstrap_error_on_significance(tau, dcf,
                                                    TAU, DCF,
                                                    tau_peak, dcf_peak,
                                                    N_pairs,
                                                    N_bootstrap=N_bootstrap_sig,
                                                    N_subset=None)
        
        # Error on significance plot
        bootstrap_error_on_significance_plot(P, dcf_sig,
                                             plot_title=plot_title_tau,
                                             plots_folder=plots_folder,
                                             file_extensions=file_extensions)
        

        # Add elements to dictionary
        r['dcf_sig_error'] = sig_error
        
        results_peaks_best.append(r)
            

    # ----------------------------------------
    # Summary file with peaks error results
    file_results_peaks_best =\
                      open(os.path.join(plots_folder,
                                        'results_peak_best_summary.txt'), 'w')
    for r in results_peaks_best:

        line = ''
        for key, value in r.iteritems():
            line += '%s %f,' % (key, value)

        line = line[:-1] + '\n'
            

        file_results_peaks_best.writelines(line)
        

    file_results_peaks_best.close()
    

    return results_peaks_best


#-------------------------------------------------------------------------------
def find_peaks_on_xcorr(tau, dcf, sigma_dcf, dcf_nelem,
                        TAU, DCF,
                        lags_range=(0.0, 0.0)):
    """ Find all the positive peaks in the cross-correlation and return a list
    for each source

    tau, dcf, n_elem_bin, perc, perc_low, perc_up

        Where perc, perc_low and perc_up are the percentiles with respect to the
    simulated distribution of cross-correlations of the CCF, CCF lower and upper
    limit.
        The lower and upper limits use the CCF - error and CCF + error

    input
    -----
    tau
    dcf
    sigma_dcf
    dcf_nelem
    TAU, DCF
    lags_range=(0.0, 0.0)    Range of time lags to search for a peak

    output
    ------
    results_peaks

    notes
    -----
    - lags_range was added to limit the search interval of cross-correlations
    and thus check their effect in the significance. If possible I will reduce
    the interval to improve the statistical constrains in cross-correlations


    Walter Max-Moerbeck, May 2, 2011
    """

    # Sets interval to search for correlaitons
    if (lags_range == None): 
        lags_range = (min(tau), max(tau))

    elif (lags_range[0] == lags_range[1]):
        lags_range = (min(tau), max(tau))


    # Find all positive peaks in the cross-correlation
    # A peak is defined as a point for which
    # dcf[i-2] < dcf[i], dcf[i-1] < dcf[i]
    # dfc[i] > dcf[i+1], dcf[i] > dcf[i+2]

    tau_peaks_i = []
    for i in arange(2, len(tau) - 2):

        if dcf[i] > dcf[i - 2] and\
           dcf[i] > dcf[i - 1] and\
           dcf[i] > dcf[i + 1] and\
           dcf[i] > dcf[i + 2] and\
           dcf[i] > 0:

            # Check if in tau range and add
            if (lags_range[0] <= tau[i]) and (tau[i] <= lags_range[1]):

                tau_peaks_i.append(i)

    # Generate info for each peak
    tau_peaks = [tau[i] for i in tau_peaks_i]
    dcf_peaks = [dcf[i] for i in tau_peaks_i]
    dcf_peaks_error = [sigma_dcf[i] for i in tau_peaks_i]

    # Calculate the significance for each peak
    tau_bins, sigma_levels, mean_level, perc_dcf_peaks =\
              confidence_intervals(TAU, DCF, tau_peaks, dcf_peaks,
                                   levels=[68.27, 95.45, 99.73],
                                   line_type=[':', '-.', '--'],
                                   line_color=['#9C0000', '#FF6F00', '#006300'])

    # Save summary of peaks
    results_peaks = []
    sig_peaks = []
    
    for i in range(len(tau_peaks)):

        # Old data structure changed to dictionary to make it easier to read
        #results_peaks.append([tau_peaks[i],
        #                      dcf_peaks[i],
        #                      dcf_peaks_error[i],
        #                      perc_dcf_peaks[1][i],
        #                      perc_dcf_peaks_ll[1][i],
        #                      perc_dcf_peaks_ul[1][i]])
    

        results_peaks.append({'tau': tau_peaks[i],
                              'dcf': dcf_peaks[i],
                              'dcf_error': dcf_peaks_error[i],
                              'dcf_sig': perc_dcf_peaks['perc'][i]})


        # To sort the peaks by significance level
        sig_peaks.append(perc_dcf_peaks['perc'][i])

    results_peaks = array(results_peaks)
    sig_peaks = array(sig_peaks)


    # Sort peaks by significance, starting with most significant
    i_sort = argsort(sig_peaks)
    i_sort = i_sort[-1::-1]
    results_peaks = results_peaks[i_sort]
    
    
    return results_peaks


#-------------------------------------------------------------------------------
def error_on_peak(ta, fa, erra,
                  tb, fb, errb,
                  delta_t,
                  min_elem_bin,
                  linear_detrending,
                  tau_peak,
                  N_sim=1000):
    """ Using RSS and FR determine the error in the cross-correlation peaks

    Use the maximum as the position of the peak. The peaks have to be previously
    determined using 

    input
    -----
    results    Summary of cross-correlation significance results

    tau_peak   Position of peak for error determination

    N_sim      Number of simulated light curve pairs for error determination

    plot_title Title for summary plot


    output
    ------
    tau_peak_mean
    tau_peak_error


    notes
    -----


    Walter Max-Moerbeck, May 18, 2011
    """


    # Calculate distribution of peak positions using RSS and FR
    TAU_PEAK = array([])

    for i in xrange(N_sim):

        # RSS and FR for both time series
        ta_s, fa_s, erra_s = simData.random_subset(ta, fa, erra)
        ta_s, fa_s, erra_s = simData.randomize_flux(ta_s, fa_s, erra_s)

        tb_s, fb_s, errb_s = simData.random_subset(tb, fb, errb)
        tb_s, fb_s, errb_s = simData.randomize_flux(tb_s, fb_s, errb_s)

        # Cross-correlate time series with same parameters
        tau_0, sigma_tau_0, dcf_0, sigma_dcf_0, n_elem_0 =\
               crosscorr(ta_s, fa_s, erra_s,
                         tb_s, fb_s, errb_s,
                         delta_t,
                         error_corr=False,
                         min_elem_bin=min_elem_bin,
                         linear_detrending=linear_detrending)
        
        # Find all peaks on DCF
        tau_peaks_all = array([])
        for i in arange(2, len(tau_0) - 2):
            
            if dcf_0[i] > dcf_0[i - 2] and\
                   dcf_0[i] > dcf_0[i - 1] and\
                   dcf_0[i] > dcf_0[i + 1] and\
                   dcf_0[i] > dcf_0[i + 2] and\
                   dcf_0[i] > 0:
                
                tau_peaks_all = concatenate((tau_peaks_all,
                                             array([tau_0[i]])))


        # Find closest one to tau_peak and save it
        i_closest = argmin(absolute(tau_peaks_all - tau_peak))
        tau_peak_closest = tau_peaks_all[i_closest]
        
        TAU_PEAK = concatenate((TAU_PEAK,
                                array([tau_peak_closest])))
        

    # Use median and percentile intervals for error and best fit
    tau_peak_median = stats.scoreatpercentile(TAU_PEAK, 50.0)
    tau_std_ll = tau_peak_median - stats.scoreatpercentile(TAU_PEAK, 15.87)
    tau_std_ul = stats.scoreatpercentile(TAU_PEAK, 84.13) - tau_peak_median
    
    # Estimate standard bootstrap error
    tau_std = std(TAU_PEAK, ddof=1)


    return tau_std, tau_peak_median, tau_std_ll, tau_std_ul, TAU_PEAK


#-------------------------------------------------------------------------------
def error_on_peak_plot(tau, dcf, sigma_dcf,
           tau_peak, tau_std, tau_peak_median, tau_std_ll, tau_std_ul, TAU_PEAK,
                       plot_title='',
                       plots_folder='',
                       file_extensions=['.png',
                                        '.pdf',
                                        '.eps'],
                       lags_range=None):
    """ Plot the cross-correlation and RSS/FR histograms for the given peak
    

    Walter Max-Moerbeck, June 15, 2011
    """
    
    # Plot data cross-correlation and RSS/FR distribution of peaks
    pylab.clf()
    pylab.subplot(211)

    pylab.errorbar(tau,
                   dcf,
                   sigma_dcf,
                   fmt='.k',
                   capsize=0)
    pylab.ylim(-1.2, 1.2)

    # set x-scale, x_lim=None by default shows all data
    if lags_range == None:
        pylab.xlim((min(tau), max(tau)))

    else:
        pylab.xlim(lags_range)
    
    x_lim = pylab.xlim()
    
    pylab.title(plot_title)
    pylab.xlabel('CCF')
    pylab.ylabel(r'$\tau$')
    
    pylab.subplot(212)

    pylab.hist(TAU_PEAK, bins=tau, color='k', histtype='step',
               label='%f +/- %f' % (tau_peak, tau_std))
    pylab.axvline(tau_peak_median, ls='-', color='k')
    pylab.axvline(tau_peak_median - tau_std_ll, ls=':', color='k')
    pylab.axvline(tau_peak_median + tau_std_ul, ls=':', color='k')

    pylab.xlim(x_lim[0], x_lim[1])

    pylab.legend(loc=0)


    # Save plots in multiple formats
    plot_xcorrsig_out =\
                      os.path.join(plots_folder,
                                plot_title.replace(' ', '_').replace('.', 'p'))

    for extension in file_extensions:

        pylab.savefig(plot_xcorrsig_out + '_peak_error' + extension)


    return


#-------------------------------------------------------------------------------
def bootstrap_error_on_significance(tau, dcf,
                                    TAU, DCF,
                                    tau_peak, dcf_peak,
                                    N_pairs,
                                    N_bootstrap=1000,
                                    N_subset=None):
    """ Bootstrap estimate of the error on the significance

    input
    -----
    tau, dcf            Data cross-correlation
    TAU, DCF            Simulation cross-correlations
    tau_peak, dcf_peak  Peak parameters
    N_bootstrap         Number of bootstrap subsets
    N_subset            Analyse a subsample of full simulation to estimate
                        error on smaller simulation

    output
    ------
    sig_median          Median of bootstrap simulations, only for reference
    sig_error           Standard deviation of bootstrap trials

    
    notes
    -----


    wmax, August 18, 2011
    """

    # Number of data points used for bootstrap analysis
    if N_subset == None:

        N_subset = N_pairs


    # Select all the random dcfs for time lag of the peak
    # Positive and negative
    i_tau_peak = where((TAU == tau_peak))[0]
    DCF_TAU_PEAK = DCF[i_tau_peak[0:N_subset]]


    # Get significance for bootstrap data sets
    P = []
    for i in xrange(N_bootstrap):
        
        # Select a list of trials to use with repetition
        index_trials = random.randint(0, N_subset, N_subset)

        # Select the data for positive dcfs
        dcfs_tau = DCF_TAU_PEAK[index_trials]

        index_pos = where(dcfs_tau >= 0)[0]
        dcfs_tau_pos = dcfs_tau[index_pos]
                         

        # Estimate significance and save result
        p = 100. * len(where(dcfs_tau_pos < dcf_peak)[0])/\
            (len(dcfs_tau_pos))

        P.append(p)
                       

    P = array(P)

    # Estimate bootstrap statistics
    sig_median = median(P)

    sig_error = std(P, ddof=1)


    return sig_median, sig_error, P


#-------------------------------------------------------------------------------
def bootstrap_error_on_significance_plot(P, dcf_sig,
                                         plot_title='',
                                         plots_folder='',
                                         file_extensions=['.png',
                                                          '.pdf',
                                                          '.eps']):
    """ Plot the estimated significance along the distribution of bootstrap
    estimates. This is to check nothing is going wrong

    wmax, August 19, 2011
    """

    # Plot distribution of bootstarp trials
    pylab.clf()

    delta_bin = 0.05
    pylab.hist(P, bins=arange(0, 100.0 + delta_bin, delta_bin),
               color='k',
               histtype='step')

    # Add significance from full simulation
    pylab.axvline(dcf_sig, color='k')

    # Add median and std estimated from bootstrap
    pylab.axvline(median(P), color='k', ls='--')
    pylab.axvline(median(P) - std(P) , color='k', ls=':')
    pylab.axvline(median(P) + std(P) , color='k', ls=':')


    pylab.xlim((min(P)- delta_bin, max(P)+ delta_bin))

    pylab.title(plot_title)
    pylab.xlabel('Significance')
    pylab.ylabel('N')
    
    # Save plots in multiple formats
    plot_xcorrsig_out =\
                      os.path.join(plots_folder,
                                plot_title.replace(' ', '_').replace('.', 'p'))

    for extension in file_extensions:

        pylab.savefig(plot_xcorrsig_out + '_sig_error' + extension)

    return

#-------------------------------------------------------------------------------
# False positive rate
#-------------------------------------------------------------------------------
def false_positive_rate_test(
    outfolder='plots_false_positives',
    N_corr=100,
    N_uncorr=100,
    ta=None,
    fa=None,
    erra=None,
    tb=None,
    fb=None,
    errb=None,
    beta_a=2.0,
    tSamp_a=1.0,
    tSim_a=365 * 10.0,
    beta_b=2.0,
    tSamp_b=1.0,
    tSim_b=365 * 10.0,
    delta_t=10.0,
    min_elem_bin=10.0,
    lags_range=(0.0, 0.0),
    file_ext=['.png', '.pdf', '.eps']):
    """ False positive rate

    Take a given sampling pattern and check the false positive rate

    Its estimates significance contours for given sampling and then simulates
    uncorrelated time series and checks the false positive rates are as
    expected.

    The p-values obtained by the Monte Carlo simlulations are only valid if 
    we test a single time lag. Here we try to estimate the false positive rate 
    when we search for a lag at an unknown time.


    input
    -----
    outfolder='plots_false_positives'
              Folder to save all the summary plots
              
    N_corr=100     Number of simualted uncorrelated data sets to find peaks 
                   and significance
    
    N_uncorr=100   Number of simulated uncorrelated data sets for significance
                   evaluation

    ta=None          Given sampling pattern to test
    fa=None    
    erra=None
    
    tb=None          Given sampling pattern to test
    fb=None
    errb=None

    beta_a=2.0           Beta for simulated light curves
    tSamp_a=1.0          Sampling period for simulated light curves
    tSim_a=365 * 10.0    Lenght of simulated light curves

    beta_b=2.0           Same as above for second time series
    tSamp_b=1.0
    tSim_b=365 * 10.0

    delta_t=10.0         Binning of time lags for cross-correlation

    min_elem_bin=10.0

    file_ext=['.png', '.pdf', '.eps']


    output
    ------


    notes
    -----
    - It searhes for peaks in cross-correlation in a given time interval. If no
    peak is found for a particular case, it sets tau=0, dcf=scipy.nan,
    perc=0. This will basically keep the count on simulated light curve pairs 
    but will count it as low significance case as no peaks was found
    
    wmax, Aug 30, 2012
    """

    #--------------------------------------------------
    # Create folder to save example simulations
    examples_folder = 'examples'
    os.mkdir(os.path.join(outfolder,
                          examples_folder))


    #--------------------------------------------------
    # Estimate significance with Monte Carlo simulation
    results =\
        crosscorr_significance_plaw_noises(ta, fa, erra,
                                           tb, fb, errb,
                                           beta_a=beta_a,
                                           tSamp_a=tSamp_a,
                                           tSim_a=tSim_a,
                                           beta_b=beta_b,
                                           tSamp_b=tSamp_b,
                                           tSim_b=tSim_b,
                                           delta_t=delta_t,
                                           N_pairs=N_uncorr,
                                           mock_test=False,
                                           Dt_proj_a=0.0,
                                           Dt_proj_b=0.0,
                                           min_elem_bin_xcorr=min_elem_bin,
                                           linear_detrending_xcorr=False,
                                           error_corr_xcorr=False,
                                           N_sim_scale_factor=1000,
                                           add_obs_noise=False)
        

    #--------------------------------------------------
    # Correlate all the sources to find peak distribution for correlated data
    index_sampling_a = array([])
    index_sampling_b = array([])

    # Arrays to save best peak results
    best_peak= {'tau': zeros(N_corr),
                'ccf': zeros(N_corr),
                'perc': zeros(N_corr)}
    
    for k in xrange(N_corr):
        
        print( 'Working on iteration %d out of %d' % (k + 1, N_corr))
        
        # Simulate uncorrelated light curves
        ta_s, sa_s, ta_sim, sa_sim, index_sampling_a =\
            simData.simulateLightCurveSampling(ta,
                                               tSamp=tSamp_a,
                                               beta=beta_a,
                                               tSim=tSim_a,
                                               real_imag_ratio=1.0e9,
                                               index_sampling=index_sampling_a)
            
        
        tb_s, sb_s, tb_sim, sb_sim, index_sampling_b =\
            simData.simulateLightCurveSampling(tb,
                                               tSamp=tSamp_b,
                                               beta=beta_b,
                                               tSim=tSim_b,
                                               real_imag_ratio=1.0e9,
                                               index_sampling=index_sampling_b)

        # Cross-correlate
        TAU, SIGMA_TAU, DCF, SIGMA_DCF, N_ELEM =\
            crosscorr(ta_s, sa_s, sa_s * 0.0,
                      tb_s, sb_s, sb_s * 0.0,
                      delta_t,
                      error_corr=False,
                      min_elem_bin=min_elem_bin,
                      linear_detrending=False)
 

        # Find most significant peak on cross-correlation and save it
        # Peaks are sorted by significanc
        results_peaks = find_peaks_on_xcorr(TAU, DCF, SIGMA_DCF, N_ELEM,
                                            results['TAU'], results['DCF'],
                                            lags_range=lags_range)

        # If no peak was found, replace it for a peak at zero with zero 
        # significance:
        if len(results_peaks) == 0:
            
            tau_peak = 0.0
            dcf_peak = nan
            perc_peak = 0.0
        
        else:
            tau_peak = results_peaks[0]['tau'] 
            dcf_peak = results_peaks[0]['dcf']
            perc_peak = results_peaks[0]['dcf_sig']  
            

        best_peak['tau'][k] = tau_peak
        best_peak['ccf'][k] = dcf_peak
        best_peak['perc'][k] = perc_peak


        #--------------------------------------------------
        # Plot examples and save 100 for later inspection
        # Plot data and cross-correlation contours
        if k < 10:

            # Data, cross-correlations and most significant peaks
            pylab.clf()
            pylab.subplot(211)
            pylab.plot(ta_s, sa_s, '.b', label='A')
            pylab.plot(tb_s, sb_s, '.r', label='B')
            pylab.xlabel('MJD')
            pylab.ylabel('Flux')
            pylab.legend(loc=0, frameon=False)
            
            pylab.subplot(212)
            pylab.plot(TAU, DCF, 'b', label='CCF')
            pylab.axvline(tau_peak, color='b')

            if not isnan(dcf_peak):
                pylab.axhline(dcf_peak, color='b')

            pylab.xlabel(r'$\tau$')
            pylab.ylabel('CCF')
            pylab.legend(loc=0, frameon=False)
            
            savefig(os.path.join(outfolder,
                                 examples_folder,
                                 '%s_%d' % ('xsamples_sim_data', k)),
                    file_ext=file_ext)


            # Significance plots, DCF
            crosscorr_significance_plaw_noises_plot(
                TAU, DCF, SIGMA_DCF,
                results['tau_bins'],
                results['sigma_levels'],
                results['mean_level'],
                results['perc_dcf'],
                plot_title='localccf_%d' % (k),
                plots_folder=os.path.join(outfolder, examples_folder),
                file_extensions=file_ext,
                y_lim=(-1.5, 1.5),
                x_lim=None,
                save_data=False)
        

    #--------------------------------------------------
    # Best peaks results significance
    pylab.clf()
    pylab.hist(best_peak['perc'],
               histtype='step',
               color='k')
    pylab.xlabel('P')
    pylab.ylabel('N')
    pylab.ylim(ymin=0.0)
        
    savefig(os.path.join(outfolder,
                         examples_folder,
                         'summary_best_peak_perc'),
            file_ext=file_ext)

    
    #--------------------------------------------------
    # Tau distribution
    # Exclude the cases where no peak is found
    index_tau = array([i_tau for i_tau in range(len(best_peak['tau'])) if\
                           not isnan(best_peak['ccf'][i_tau])])
    
    pylab.clf()
    pylab.hist(best_peak['tau'][index_tau],
               histtype='step',
               color='k')
    pylab.xlabel(r'$\tau$')
    pylab.ylabel('N')
    pylab.ylim(ymin=0.0)
        
    savefig(os.path.join(outfolder,
                         examples_folder,
                         'summary_best_peak_tau'),
            file_ext=file_ext)
   

    #--------------------------------------------------
    # Cumulative distribution of p-values
    p_values = 1.0 - best_peak['perc'] / 100.0

    pylab.clf()
    pylab.hist(p_values,
               histtype='step',
               normed=True,
               cumulative=True,
               color='k',
               bins=100,
               range=(0.0, 1.0))
    pylab.xlabel('p-value')
    pylab.ylabel('N')
    pylab.ylim(ymin=0.0)
    
    savefig(os.path.join(outfolder,
                         examples_folder,
                         'summary_best_peak_perc'),
            file_ext=file_ext)
    
    #--------------------------------------------------
    # Correction factor to make distribution uniform
    # Estimate the cumulative distribution on p_values first
    N = len(p_values) * 1.0
    delta_p = 1.e-6 
    p_x = arange(delta_p, 1.0 + delta_p, delta_p)
    p_cumulative = array([len(where(p_values <= p_value)[0]) / N for\
                              p_value in p_x])

    pylab.clf()
    pylab.subplot(211)
    pylab.plot(p_x, p_cumulative, color='k')
    pylab.xlabel('p-value')
    pylab.ylabel('fraction')

    pylab.subplot(212)
    pylab.plot(p_x, p_cumulative / p_x, color='k')
    pylab.xlabel('p-value')
    pylab.ylabel('correction')

    # Save figure
    savefig(os.path.join(outfolder,
                         examples_folder,
                         'summary_best_peak_cumulative_correction'),
            file_ext=file_ext)

    #--------------------------------------------------
    # Plot again but zoomed to better see high significance cases
    pylab.clf()
    pylab.subplot(211)
    pylab.plot(p_x, p_cumulative, color='k')
    pylab.xlabel('p-value')
    pylab.ylabel('fraction')
    pylab.xlim((0.0, 0.1))
 
    pylab.subplot(212)
    pylab.plot(p_x, p_cumulative / p_x, color='k')
    pylab.xlabel('p-value')
    pylab.ylabel('correction')
    pylab.xlim((0.0, 0.1))

    # Save figure
    savefig(os.path.join(outfolder,
                         examples_folder,
                         'summary_best_peak_cumulative_correction_zoom'),
            file_ext=file_ext)


    # Save figure data
    file_pvalue_cum_data =\
        os.path.join(outfolder,
                     examples_folder,
                     'pvalues_cumulative_data.txt')
        
    header = '# Data p-values cumulative distribution\n'
    header += '# p_value, n_cumulative\n'
    
    savefile(file_pvalue_cum_data,
             data=transpose((p_x, p_cumulative)),
             fmt='%f',
             header=header)

    # correction    
    file_pvalue_correction_data =\
        os.path.join(outfolder,
                     examples_folder,
                     'pvalues_correction_data.txt')
    
    header = '# Data p-values correction\n'
    header += '# p_value, f_correction\n'
    
    savefile(file_pvalue_correction_data,
             data=transpose((p_x, p_cumulative / p_x)),
             fmt='%f',
             header=header)


    #------------------------------------------------------------
    # Save pickle file with best_peak, to be able to reanalyze data if 
    # necessary
    file_pickle = os.path.join(outfolder,
                               examples_folder,
                               'data.dat')
    fileObj = open(file_pickle, 'w')

    pickle.dump(best_peak, fileObj)
            
    fileObj.close()


    return best_peak

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------
def crosscorr_test(d=10):
    """ A few test for crosscorr() function

    Walter Max-Moerbeck, April 20, 2010
    """
    
    #--------------------------------------------------
    # Test with a pulse of with 0.5 at center of sequence and noise
    ta = arange(100)
    fa = 0.1 * random.randn(100)
    fa[45:47] = ones(2)  
    erra = 0.1 * ones(100)

    tb = arange(100)
    fb = 0.1 * random.randn(100)
    fb[45 + d:47 + d] = ones(2)    
    errb = 0.1 * ones(100)

    TAU, SIGMA_TAU, DCF, SIGMA_DCF, N_ELEM = crosscorr(ta, fa, erra,
                                                       tb, fb, errb,
                                                       delta_t=1,
                                                       error_corr=False)

    pylab.subplot(311)
    pylab.errorbar(ta, fa, erra)
    pylab.errorbar(tb, fb, errb)

    pylab.subplot(312)
    pylab.errorbar(TAU, DCF, SIGMA_DCF, SIGMA_TAU)

    pylab.subplot(313)
    pylab.plot(TAU, N_ELEM)
    pylab.show()

    
    return

#-------------------------------------------------------------------------------
def autocorr_test():
    """ A few test for autocorr() function
    
    Walter Max-Moerbeck, April 20, 2010
    """
    
    #--------------------------------------------------
    # Test with a pulse of with 0.5 at center of sequence and noise
    ta = arange(100)
    fa = 0.1 * random.randn(100)
    fa[45:47] = ones(2)  
    erra = 0.1 * ones(100)

    TAU, SIGMA_TAU, DCF, SIGMA_DCF, N_ELEM = autocorr(ta, fa, erra,
                                                      delta_t=1,
                                                      error_corr=False)

    pylab.subplot(311)
    pylab.errorbar(ta, fa, erra)

    pylab.subplot(312)
    pylab.errorbar(TAU, DCF, SIGMA_DCF, SIGMA_TAU)

    pylab.subplot(313)
    pylab.plot(TAU, N_ELEM)
    pylab.show()

    return
