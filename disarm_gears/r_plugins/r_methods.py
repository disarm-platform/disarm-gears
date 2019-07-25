import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

# Load R packages
rstats = importr('stats')
rmgcv = importr('mgcv')
pandas2ri.activate()
rbase = importr('base')
#rutils = importr('utils')


def pdframe2rdframe(data):
    '''
    Converts a pandas DataFrame into an R dataframe

    :param data: Data to convert into an R dataframe
                 pandas DataFrame
    :return: robjects.vectors.DataFrame
    '''
    # This step is just to make sure pandas2ri.DataFrame works as expected
    assert isinstance(data, pd.DataFrame)
    data2 = pd.DataFrame({ci: [vi for vi in data[ci]] for ci in data.columns})
    return pandas2ri.DataFrame(data2)


def mgcv_fit(formula, data, family='gaussian', weights=None, method='REML', bam=False, chunk_size=1000):
    '''
    Fit a Generalized Additive Model

    This function is a wrapper of the MGCV package in R.
    See its definition for more details https://www.rdocumentation.org/packages/mgcv/versions/1.8-28/topics/gam

    :param formula: R formula
                    string
    :param data: Data to fit the model to
                 pandas DataFrame
    :param family: Noise model distribution
                   string. One of 'gaussian', 'binomial', 'poisson' (default 'gaussian').
    :param weights: Observation weights
                    numpy ndarray
    :param method: Smoothing parameter estimation method (see R MGCV documentation)
                   string, one of 'GCV.Cp', 'GACV.Cp', 'REML', 'P-REML', 'ML', 'fREML' (default 'REML').
    :param bam: If MGCV implementation for large datasets should be used
                Boolean
    :param chunk_size: Size of chunks in which the model matrix is created (Only used when bam=True).
                       Integer
    :return: robjects.vectors.ListVector
    '''
    assert isinstance(data, pd.DataFrame)

    # Define family
    if family == 'gaussian':
        rfamily = rstats.gaussian(link='identity')
    elif family == 'binomial':
        rfamily = rstats.binomial(link='logit')
    elif family == 'poisson':
        rfamily = rstats.poisson(link='log')
    else:
        raise NotImplementedError

    rdata = pdframe2rdframe(data)
    rformula = robjects.Formula(formula)

    if weights is None:
        if not bam:
            gam = rmgcv.gam(formula=rformula, data=rdata, family=rfamily, method=method)
        else:
            gam = rmgcv.bam(formula=rformula, data=rdata, family=rfamily, method=method, chunk_size=chunk_size)
    else:
        #TODO assert weights
        if not bam:
            gam = rmgcv.gam(formula=rformula, data=rdata, family=family, weights=weights, method=method)
        else:
            raise NotImplementedError

    return gam


def mgcv_predict(gam, data, response_type='response'):
    '''
    Make predictions using a fitted GAM model

    See R MGCV package

    :param gam: A gam model previously fitted
                robjects.vectors.ListVector
    :param data: Input data points where predictions are made
                 pandas DataFrame
    :param response_type: Space or transformation in which the prediction is returned (see R MGCV documentation)
                          string. One of 'response', 'link', 'lpmatrix'.
    :return: numpy ndarray of shape (n_data, ) with the posterior mean values.
    '''

    assert isinstance(gam, robjects.vectors.ListVector)
    assert isinstance(data, pd.DataFrame)
    if response_type not in ['link', 'response', 'lpmatrix']:
        raise NotImplementedError
    rdata = pdframe2rdframe(data)
    return np.array(rmgcv.predict_gam(gam, newdata=rdata, type=response_type))


def mgcv_posterior_samples(gam, data, n_samples=100, response_type='inverse_link') -> np.ndarray:
    '''
    Generate samples from the posterior distribution of a GAM

    See R MGCV package

    :param gam: A gam model previously fitted
                robjects.vectors.ListVector
    :param data: Input data points where simulations are generated
                 Pandas DataFrame
    :param n_samples: Number of samples to generate
                      Integer (default 100)
    :param response_type: Space or transformation in which the prediction is returned (see R MGCV documentation)
                          String. One of 'response', 'link', 'inverse_link', 'lpmatrix'.
    :return: numpy ndarray of shape (n_samples, n_data)
    '''

    assert isinstance(gam, robjects.vectors.ListVector)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(n_samples, int)
    assert n_samples > 0

    gam_coef = np.array(rstats.coef(gam))
    gam_vcov = np.array(rstats.vcov(gam))
    M = np.random.multivariate_normal(gam_coef, gam_vcov, size=n_samples)
    rdata = pdframe2rdframe(data)
    LP = np.array(rmgcv.predict_gam(gam, newdata=rdata, type='lpmatrix'))
    _post = np.dot(M, LP.T)

    if response_type == 'link':
        samples = _post

    elif response_type == 'inverse_link':
        family, link = get_family(gam), get_link(gam)
        if family == 'gaussian':
            assert link == 'identity'
            samples = _post
        elif family == 'binomial':
            assert link == 'logit'
            samples = 1. / (1. + np.exp(-_post))
        else: #family == 'poisson'
            assert link == 'log'
            samples = np.exp(_post)

    elif response_type == 'response': #TODO the results accuracy of this case have not been tested yet
        family, link = get_family(gam), get_link(gam)
        if family == 'gaussian' and link == 'identity':
            ix_scale = get_names(gam).index('scale') #TODO check this line
            sigma_noise = np.sqrt(gam[ix_scale])
            samples = _post + np.random.normal(0, sigma_noise, _post.size).reshape(_post.shape)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return samples


def get_family(gam):
    '''
    Get family of a GAM

    :param gam: A gam model previously fitted
                robjects.vectors.ListVector

    :return: string
    '''
    return gam[3][0][0]


def get_link(gam):
    '''
    Get link function of a GAM

    :param gam: A gam model previously fitted
                robjects.vectors.ListVector

    :return: string
    '''
    return gam[3][1][0]


def get_names(obj):
    '''
    Get name of an R object

    :param obj: Object to extract the names from
                R object
    :return: List of names
    '''
    obj_names = rbase.names(obj)
    return list(obj_names)


def summary(model):
    '''Print R model summary'''
    print(rbase.summary(model))


def mgcv_get_rho_power_exp2(iter_formula, data, smooth_dim, family='gaussian', weights=None, method='REML'):
    '''
    For a GAM model with gp smooth of type exponentiated quadratic find the best lengthscale (rho) parameter

    :param iter_formula: R formula
                         String
    :param data: Data to fit the model to
                 pandas DataFrame
    :param smooth_dim: Dimensionality of the smooth interpolation. The only values accepted are 1 and 2.
                 Integer
    :param family: Noise model distribution (see mgcv_fit)
                   String
    :param weights: Observation weights (see mgcv_fit)
                    Numpy ndarray
    :param method: Smoothing parameter estimation method (see mgcv_fit).
                   String, defaults to 'REML'.
    :return: float
    '''

    # Fit initial model and check model components are in place
    m0 = mgcv_fit(iter_formula %'', data=data, family=family, weights=weights, method=method)
    ix_gcv = get_names(m0).index('gcv.ubre')
    ix_smooth = get_names(m0).index('smooth')
    if smooth_dim == 1:
        raise NotImplementedError
        #ix_gpdefn = get_names(m0[ix_smooth][0]).index('gp.defn')
        #rho0 = m0[ix_smooth][0][ix_gpdefn][1]
    elif smooth_dim == 2:
        ix_gpdefn = get_names(m0[ix_smooth][0][0][0]).index('gp.defn')
        rho0 = m0[ix_smooth][0][0][0][ix_gpdefn][1]
    else: #elif smooth_dim == 3:
        raise NotImplementedError
        #ix_gpdefn = get_names(m0[ix_smooth][0][0][0]).index('gp.defn')
        #rho0 = m0[ix_smooth][0][0][0][ix_gpdefn][1]
        #rho1 = m0[ix_smooth][0][0][1][ix_gpdefn][1]

    # Grid of values to try
    if rho0 >= 8:
        rho_grid = [rho0/(2 ** i) for i in range(int(np.log2(rho0)))[::-1]]
    else:
        rho_grid = [i * rho0 / 10. for i in range(1, 11, 2)]

    # Lambda function to fit all models
    iter_fit = lambda rho_i:  mgcv_fit(iter_formula %', %s, 2' %rho_i,
                                       data=data, family=family, weights=weights, method=method)
    gcv_vals = np.hstack([iter_fit(rho_i)[ix_gcv] for rho_i in rho_grid])

    return rho_grid[np.argmin(gcv_vals)]

