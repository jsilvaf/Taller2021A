"""
OVRO 40m program variability
 
Various utility routines
 
Walter Max-Moerbeck, 16 April 2010
"""

from math import ceil, floor

from scipy import *
from scipy.optimize import leastsq

import pylab

#-------------------------------------------------------------------------------
# Binning
#-------------------------------------------------------------------------------
def bin_data(t, f, delta_t,
             positive=True,
             min_elem_bin=2):
    """ Bin data and calculate error associated with each bin as the dispersion
    on the given bin if required.

    INPUT
    -----
    t              Time in days for time series
    f              Flux
    delta_t        Width of the bins

    positive=True  Lower limit is 0
    positive=False Lower limit is given by data. There is always a bin
                   centered in zero.

    min_elem_bin   Minimum number of elements per bin

    return_n_elem  True: Return number of elements per bin


    OUTPUT
    ------
    T         Center of the bin
    DELTA_T   Half width of the bin
    F         Sample mean for the bin
    SIGMA_F   Standard error on the mean
    n_elem    Number of elements per bin

    NOTES
    -----
    This routine was originally written for use with Structure Functions 
    and cross-correlation functions. It follows the idea of Edelson and Krolik
    1988 paper.
    
    The error estimate used is just a measure of the dispersion. For serious 
    uses better error measurements might be considered.
    """

    # Define constants used often
    delta_t_div_2 = delta_t / 2

    #--------------------------------------------------
    # Find extreme times on data set. If the sequence is empty return
    # empty arrays
    index_sort = argsort(t)
    t = t[index_sort]
    f = f[index_sort]
    
    try:
        tmax = t[-1]
        tmin = t[0]

    except IndexError:
        T = array([])
        DELTA_T = array([])
        
        F = array([])
        SIGMA_F = array([])

        N_ELEM = array([])

        return T, DELTA_T, F, SIGMA_F, N_ELEM


    #----------------------------------------
    # Create an array of bin centers
    # Lower limit is zero
    if positive == True:
        
        # find index of maximum bin
        N = int(ceil(tmax / delta_t - 1.0))

        # array of bin centers
        bin_center = arange(0., N + 1.0) * delta_t + delta_t_div_2
        
    # Lower limit given by data and always a bin centered on zero
    elif positive == False:
    
        # find maximum bin index
        N = int(ceil(tmax / delta_t - 0.5))
    
        # find minimum bin index
        M = int(floor(tmin / delta_t + 0.5))
    
        # array of bin centers
        bin_center = arange(M, N + 1.0) * delta_t
    

    #----------------------------------------
    # Initialize arrays to store results
    N_bins = len(bin_center)
        
    T = zeros(N_bins)
    DELTA_T = ones(N_bins) * delta_t_div_2
        
    F = zeros(N_bins)
    SIGMA_F = zeros(N_bins)

    N_ELEM = zeros(N_bins)
            
    #----------------------------------------
    # bin the data
    # Start looking at first element
    i = 0    # t, f index
    k = 0    # non-empty bin index
    N_elem = len(t)
    for center in bin_center:
        
        # find the elements on this bin, we don't need lower limit
        ul = center + delta_t_div_2

        # Initial variables for this bin
        N_elem_bin = 0
        sum = 0.0
        sum2 = 0.0

        # move on array
        while i < N_elem:

            # If we are here, we know ll <= t[i]
            # Here we find elements in the time bin
            if t[i] < ul:

                f_i = f[i]

                sum += f_i
                
                sum2 += f_i * f_i

                N_elem_bin += 1

                i += 1                
            
            # If we got here ul < t[i]
            # We go for next time bin for this same t
            else:
                break


        # Check it has the minimum number of elements and save
        if N_elem_bin >= min_elem_bin:

            # Compute mean and std
            F[k] = sum / N_elem_bin
            SIGMA_F[k] =\
                      sqrt((sum2 - sum * sum / N_elem_bin)) / (N_elem_bin - 1.0)
                        
            T[k] = center
            # Was defined like this above
            # DELTA_T[k] = delta_t_div_2
            
            N_ELEM[k] = N_elem_bin

            k += 1
            

    #----------------------------------------
    # cut off the unused bins. These are produced by bins with less than 
    # minimum number of elements
    T = T[:k]
    DELTA_T = DELTA_T[:k]
        
    F = F[:k]
    SIGMA_F = SIGMA_F[:k]

    N_ELEM = N_ELEM[:k]
    

    return T, DELTA_T, F, SIGMA_F, N_ELEM


#-------------------------------------------------------------------------------
def bin_data_original(t, f, delta_t,
                      positive=True,
                      min_elem_bin=2):
    """ Bin data and calculate error associated with each bin as the dispersion
    on the given bin if required.

    INPUT
    -----
    t              Time in days for time series
    f              Flux
    delta_t        Width of the bins

    positive=True  Lower limit is 0
    positive=False Lower limit is given by data. There is always a bin
                   centered in zero.

    min_elem_bin   Minimum number of elements per bin

    return_n_elem  True: Return number of elements per bin


    OUTPUT
    ------
    T         Center of the bin
    DELTA_T   Half width of the bin
    F         Sample mean for the bin
    SIGMA_F   Standard error on the mean
    n_elem    Number of elements per bin

    NOTES
    -----
    This routine was originally written for use with Structure Functions 
    and cross-correlation functions. It follows the idea of Edelson and Krolik
    1988 paper.
    
    The error estimate used is just a measure of the dispersion. For serious 
    uses better error measurements might be considered.
    """

    #--------------------------------------------------
    # Find extreme times on data set. If the sequence is empty return
    # empty arrays
    try:
        tmax = max(t)
        tmin = min(t)

    except ValueError:
        T = []
        DELTA_T = []
        
        F = []
        SIGMA_F = []

        N_ELEM = []

        return array(T), array(DELTA_T), array(F), array(SIGMA_F), array(N_ELEM)

    #----------------------------------------
    # Create an array of bin centers

    # Lower limit is zero
    if positive == True:
        
        # find index of maximum bin
        N = int(ceil(tmax / delta_t - 1.0))

        # array of bin centers
        bin_center = arange(0., N + 1.0) * delta_t + delta_t / 2.0
        
    # Lower limit given by data and always a bin centered on zero
    elif positive == False:
        
        # find maximum bin index
        N = int(ceil(tmax / delta_t - 0.5))

        # find minimum bin index
        M = int(floor(tmin / delta_t + 0.5))
        
        # array of bin centers
        bin_center = arange(M, N + 1.0) * delta_t

    #----------------------------------------
    # Initialize arrays to store results
    N_bins = len(bin_center)
        
    T = zeros(N_bins)
    DELTA_T = zeros(N_bins)
        
    F = zeros(N_bins)
    SIGMA_F = zeros(N_bins)

    N_ELEM = zeros(N_bins)
            
    #----------------------------------------
    # bin the data
    # from first to penultimate bin
    k = 0
    for center in bin_center[:-1]:
        
        # find the elements on this bin
        ll = center - delta_t / 2.0
        ul = center + delta_t / 2.0
        index = where((ll <= t) & (t < ul))[0]
        
        # check that there are more than minimum number of elements per bin
        if len(index) >= min_elem_bin:
            
            # estimate the values
            mean_f = mean(f[index])
            sigma_f = sqrt( sum((f[index] - mean_f) ** 2.0)) /\
                                (len(index) - 1)
        
            # save results on arrays and advance counter
            F[k] = mean_f
            SIGMA_F[k] = sigma_f

            T[k] = center
            DELTA_T[k] = delta_t / 2.0

            N_ELEM[k] = len(index)

            k += 1

    # last bin -------------------
    # find the elements on this bin
    center = bin_center[-1]
    ll = center - delta_t / 2.0
    ul = center + delta_t / 2.0    
    index = where((ll <= t) & (t <= ul))[0]
        
    # check that there are more than minimum number of elements per bin
    if len(index) >= min_elem_bin:
        
        # estimate the values
        mean_f = mean(f[index])
        sigma_f = sqrt( sum((f[index] - mean_f) ** 2.0)) /\
                            (len(index) - 1)

        # save results on arrays and advance counter
        F[k] = mean_f
        SIGMA_F[k] = sigma_f

        T[k] = center
        DELTA_T[k] = delta_t / 2.0

        N_ELEM[k] = len(index)

        k += 1


    #----------------------------------------
    # cut off the unused bins. These are produced by bins with less than 
    # minimum number of elements
    T = T[:k]
    DELTA_T = DELTA_T[:k]
        
    F = F[:k]
    SIGMA_F = SIGMA_F[:k]

    N_ELEM = N_ELEM[:k]


    return T, DELTA_T, F, SIGMA_F, N_ELEM


#-------------------------------------------------------------------------------
def reduced_chisq(y, dy, ym, dof=1):
    """ Chi square

    Input
    -----
    y, dy    Data and error
    ym       Model
    dof=1    Number of degrees of freedom
    
    Output
    ------
    redchisq Reduced chi square

    Walter Max-Moerbeck, August 14, 2009
    """

    return sum(((y - ym) / dy)**2) / dof


# --------------------------------------------------------------------------
def lin_lsq_fit(x, y, sigma_y, redchisq=False):
    """ Least-square fit to a straight line y = a + b * x
    Based on Bevington & Robinson, 2003. pg 114.

    Input
    -----
    x, y, sigma_y    x,y data and error
    redchisq=False   Return reduced chisq of the fit

    Output
    ------
    a, sigma_a       Position coefficient and error
    b, sigma_b       Slope and error
    """
    
    # Auxiliar quantities
    Sxx = sum(x**2 / sigma_y**2)
    Sxy = sum(x*y / sigma_y**2)
    Sx  = sum(x / sigma_y**2)
    Sy  = sum(y / sigma_y**2)
    Ss  = sum(1.0 / sigma_y**2)
    Delta = Ss * Sxx - Sx**2

    # Fit parameters
    a = (Sxx * Sy - Sx * Sxy) / Delta
    b = (Ss * Sxy - Sx * Sy) / Delta
    
    # Parameters error
    sigma_a = sqrt(Sxx / Delta)
    sigma_b = sqrt(Ss / Delta)
    
    # calculate reduced \chi^2 if required and returns
    if redchisq == True:
        dof = len(x) - 2
        redchisq = sum((y - (a + b * x))**2 / sigma_y**2) / dof
        
        return a, b, sigma_a, sigma_b, redchisq, dof


    return a, b, sigma_a, sigma_b


#-------------------------------------------------------------------------------
def detrend_linear(x, y, sigma_y=None):
    """ Remove linear tren from data by substrasting a linear fit

    Walter Max-Moerbeck, January 26, 2011
    """
    
    # Use uniform weights if None is given
    if sigma_y == None:
        sigma_y = ones(len(y))
        
    # Linear fit
    a, b, sigma_a, sigma_b = lin_lsq_fit(x, y, sigma_y, redchisq=False)

    # Detrend
    y_det = y - (a + b * x)
    
    return y_det


# --------------------------------------------------------------------------
def poly_fit_error(x, y, dy, N):
    """ Fit a N-th order polynonial to the data 
    y = p(n) + p(n-1)*x + p(n-2)*x**2 + ... + p(0)*x**n
    
    Use Bevington & Robinson matrix solution
    """

    # Initialize the beta and alpha matrices
    beta = zeros((1, N + 1))
    alpha = zeros((N + 1, N + 1))

    # Calculate the elements
    for k in range(0, N + 1):

        beta[0][k] = sum(y * x**(N - k) / dy**2)
        
        for l in range(0, N + 1):

            alpha[l][k] = sum(x**(N - l) * x**(N - k) / dy**2)

    # Take the inverse of alpha, error or cavariance matrix
    alpha_inv = linalg.inv(alpha)

    # Return the parameters
    p = dot(beta, alpha_inv)[0]

    # Evaluate model
    ym = polyval(p, x)

    # Get redchisq
    redchisq = sum( (y - ym)**2 / dy**2 ) / (size(x) - N - 1)

    return p, alpha_inv, redchisq


# --------------------------------------------------------------------------
def polyval_error(p, x, covar):
    """ Evaluate the polynomial in x calculating the errors from 
    the covariance matrix

    y = p(n) + p(n-1)*x + p(n-2)*x**2 + ... + p(0)*x**n

    Use matrix version of Bevignton and Robinson solution
    """
    
    # Degree of polynomial
    N = size(pr) - 1

    # Evaluate the polynomial
    y = polyval(p, x)
    
    # Initialize array of errors
    dy = zeros(size(x))

    # For each data point evaluate and the error
    for i in range(0, size(x)):

        # Evaluate the derivatives with respect to parameters at x
        dy_dp= [x[i]**(N - j) for j in range(0, N + 1)]

        # Evaluate the error
        dy[i] = sqrt( dot(dy_dp, dot(covar, dy_dp)) ) 

    return y, dy


# --------------------------------------------------------------------------
def poly_norm(p, xl, xu):
    """ Normalize to the maximum value a polynomial with coefficients p. 
    Look for a maximum on the interval [xmin, xmax].
    Return the coefficients of the normalized polynomial
    """
    
    # Degree of polynomial
    N = size(p) - 1

    # Get the coefficients of the derivative polynomial
    p_dev = [p[i]*(N - i) for i in range(0, N)]

    # Extrema of polynomial are roots of derivative
    r = roots(p_dev)
    
    # Find the root on desired interval
    for j in range(0, size(r)):
        
        if isreal(r[j]) and (r[j] > xl and r[j] < xu):
            xmax = real(r[j])

    # Get the value of polynomial at that point and normalize
    ymax = polyval(p, xmax)
    p_norm = p / ymax

    return p_norm, xmax, ymax


#-------------------------------------------------------------------------------
def gauss_fit(x, y, dy, p0):
    """ Fit a Gaussian function to the data.
    Use dy (sigma_y) to weight the data points

    Input
    -----
    x, y, dy     Data set
    p            Gaussian parameters
                 p[0] Amplitud
                 p[1] Mean
                 p[2] Standard deviation

    Output
    ------
    A, mu, sigma     Least square fit parameters
    cov_mat          Covariance matrix
    redchisq         Reduced chi square of the fit

    Note
    ----
    Adapted from scipy.optimize cookbook
    http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    
    Walter Max-Moerbeck, June 3, 2009
    """

    def residuals(p, x, y, dy):
        A, mu, sigma = p
        err = (y - A * exp(-0.5 * ((x - mu) / sigma) ** 2)) / dy
        return err

    output = leastsq(residuals, p0, args=(x, y, dy), full_output=1)

    # fit parameters
    A = output[0][0]
    mu = output[0][1]
    sigma = output[0][2]

    # reduced chi square
    ym = A * exp(-0.5 * ((x - mu) / sigma) ** 2)
    redchisq = sum(((y - ym)/dy)**2) / (size(x) - size(p0))

    # covariance matrix
    cov_mat = output[1] * redchisq
    
    return A, mu, sigma, cov_mat, redchisq


#-------------------------------------------------------------------------------
def gauss_fit_1(x, y, dy, p0):
    """ Fit a Gaussian function to the data.
    Use dy (sigma_y) to weight the data points

    Input
    -----
    x, y, dy     Data set
    p            Gaussian parameters
                 p[0] Amplitud
                 p[1] Mean
                 p[2] Standard deviation

    Output
    ------
    A, mu, sigma     Least square fit parameters
    cov_mat          Covariance matrix
    redchisq         Reduced chi square of the fit

    Note
    ----
    - Adapted from scipy.optimize cookbook
    http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

    - The final goal is to addapt this to be a general fitting function
    
    Walter Max-Moerbeck, June 3, 2009
    """

    def residuals(p, x, y, dy):
        a, b, c = p
        err = (y - (a * x**2 + b * x + c)) / dy
        return err    

    output = leastsq(residuals, p0, args=(x, y, dy), full_output=1)

    # fit parameters
    a = output[0][0]
    b = output[0][1]
    c = output[0][2]

    # reduced chi square
    ym = a * x**2 + b * x + c
    redchisq = sum(((y - ym)/dy)**2) / (size(x) - size(p0))

    # covariance matrix
    cov_mat = output[1] * redchisq
    
    return a, b, c, cov_mat, redchisq


#-------------------------------------------------------------------------------
# Plots
#-------------------------------------------------------------------------------
def savefig(plotname, file_ext=['.png']):
    """ Save a figure in multiple formats

    wmax, Feb 14, 2012
    """

    for ext in file_ext:

        if ext[0] == '.':
            pylab.savefig(plotname + ext)
            
        else:
            pylab.savefig(plotname + '.' + ext)

    
    return


#-------------------------------------------------------------------------------
# Utilities to save and read files
#-------------------------------------------------------------------------------
def savefile(file_name, data=([], []), fmt='', header='# File description\n'):
    """ Save a list of arrays of the same size on a file adding a header

    Walter Max-Moerbeck, February 8, 2011
    """

    # Save data without header
    savetxt(file_name, data, fmt=fmt)

    # Add file and append header
    file = open(file_name, 'r')
    lines = file.readlines()
    file.close()
    
    file = open(file_name, 'w')
    file.writelines(header)
    file.writelines(lines)
    file.close()
    
    return


#-------------------------------------------------------------------------------
def read_csvfile(file,
                 delimiter=',',
                 string_columns = []):
    """ Read a csv file where the first line has the names of the fields and
    the rest are data

    Walter Max-Moerbeck, August 8, 2011
    """

    fileObj = open(file, 'r')

    first_line = True
    for line in fileObj:

        if first_line == True:
            keys = line.strip().strip('#').split(',')

            data = {}
            for key in keys:
                data[key] = []

            first_line = False

        else:

            fields = line.strip().split(delimiter)

            for i in xrange(len(fields)):

                if not i in string_columns: 
                    data[keys[i]].append(float(fields[i]))
            
                else:
                    data[keys[i]].append(fields[i])
       

    # Convert to scipy array
    for key in keys:
        data[key] = array(data[key])

    fileObj.close()
     
    return data


#-------------------------------------------------------------------------------
def read_csvfile_simple(file,
                        delimiter=',',
                        string_columns = [],
                        comments=['#'],
                        keys=['time', 'flux', 'error']):
    """ Read a csv file where the first line has the names of the fields and
    the rest are data

    Walter Max-Moerbeck, June 28, 2012
    """

    fileObj = open(file, 'r')

    # Create data struture to save data
    data = {}
    for key in keys:
        data[key] = []

    # Read file
    for line in fileObj:
        
        # Skip comment lines
        if line[0] in comments:
            continue

        # Read data from non comment lines
        fields = line.strip().split(delimiter)
    
        for i in xrange(len(fields)):
                
            if not i in string_columns: 
                data[keys[i]].append(float(fields[i]))
                    
            else:
                data[keys[i]].append(fields[i])
        

    # Convert to scipy array
    for key in keys:
        data[key] = array(data[key])

    fileObj.close()

    return data


#-------------------------------------------------------------------------------
def save_to_radio_flux_file(time, flux, error, file_name):
    """ Save to .csv file the data on a given source
    
    Walter Max-Moerbeck, November 17, 2010
    """
    
    file = open(file_name, 'w')

    # Print comments
    line = 'mjd,flux,flux_err\n'
    file.write(line)

    # write data to file
    for i in range(len(time)):
        
        line = '%f,%f,%f\n' % (time[i], flux[i], error[i])

        file.write(line)

    file.close()

    return


#-------------------------------------------------------------------------------
# Utilities to plot analysis results
#-------------------------------------------------------------------------------
def plot_data(time, flux, error,
              flux_legend='Jy',
              time_legend='MJD',
              x_lim=(0.0, 0.0),
              file_name='out_data.pdf'):
    """ Plot data for a given band
    
    Walter Max-Moerbeck, June 28, 2012
    """
    
    pylab.clf()
    pylab.errorbar(time,
                   flux,
                   error,
                   fmt='.k',
                   capsize=0)
    
    pylab.xlabel(time_legend)
    pylab.ylabel(flux_legend)
    
    if x_lim[0] != x_lim[1]:
        pylab.xlim(x_lim)
        
        pylab.ylim(ymin=0)
    
    pylab.savefig(file_name)
    
    return

#-------------------------------------------------------------------------------
def plot_psd_and_fit(f, P, 
                     f_mean, psd_mean, psd_error,
                     f_units='1/day',
                     P_units='P',
                     file_name='out_psd.pdf'):
    """ Plot the periodogram along with best fit

    wmax, June 28, 2012
    """

    pylab.clf()

    pylab.plot(f, P, '.-k')
    pylab.errorbar(f_mean, psd_mean, psd_error, fmt='.k')
    
    pylab.xlabel(f_units)
    pylab.ylabel(P_units)

    pylab.savefig(file_name)

    return

#-------------------------------------------------------------------------------
def plot_fit_summary(betas, P_values,
                     file_name='fit_summary.pdf'):
    """ Plot fit summary

    wmax, June 28, 2012
    """

    pylab.clf()
    pylab.plot(betas, P_values, '.-k')

    pylab.ylim((0.0, 1.0))

    pylab.xlabel(r'$\beta$')
    pylab.ylabel('P')

    pylab.savefig(file_name)

    return


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------
def bin_data_test():
    """ Testing for bin_data()

    Walter Max-Moerbeck, April 16, 2010
    """
    
    #--------------------------------------------------
    # Test with a pulse of with 0.5 at center of sequence and noise
    t = arange(100)
    f = 0.1 * random.randn(100)
    f[45:55] = ones(10)


    T, D_T, F, D_F, N_ELEM = bin_data(t, f, delta_t=0.5,
                                      positive=True,
                                      min_elem_bin=1)
    pylab.subplot(311)
    pylab.plot(t, f)
    
    pylab.subplot(312)
    pylab.errorbar(T, F, D_F, D_T)

    pylab.subplot(313)
    pylab.plot(T, N_ELEM)
    pylab.show()

    
    #--------------------------------------------------
    # Smooth to two days
    T, D_T, F, D_F, N_ELEM = bin_data(t, f, delta_t=10.0,
                                      positive=True,
                                      min_elem_bin=1)
    pylab.subplot(311)
    pylab.plot(t, f)
    
    pylab.subplot(312)
    pylab.errorbar(T, F, D_F, D_T)

    pylab.subplot(313)
    pylab.plot(T, N_ELEM)
    pylab.show()

    return
