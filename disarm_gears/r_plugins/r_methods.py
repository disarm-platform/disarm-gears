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
    :return: R object
    '''
    # This step is just to make sure pandas2ri.DataFrame works as expected
    data2 = pd.DataFrame({ci: [vi for vi in data[ci]] for ci in data.columns})
    return pandas2ri.DataFrame(data2)

def mgcv_fit(formula, data, family='gaussian', weights=None, method="REML"):
    '''
    :param formula:
                    String.
    :param data:
                 pandas DataFrame
    :param family:
    :param method:
    :return:
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
        gam = rmgcv.gam(formula=rformula, data=rdata, family=rfamily, method=method)
    else:
        gam = rmgcv.gam(formula=rformula, data=rdata, family=family, weights=weights, method=method)
    return gam


def mgcv_predict(gam, data, response_type='response'):

    assert isinstance(gam, robjects.vectors.ListVector)
    assert isinstance(data, pd.DataFrame)
    if response_type not in ['link', 'response', 'lpmatrix']:
        raise NotImplementedError
    rdata = pdframe2rdframe(data)
    return np.array(rmgcv.predict_gam(gam, newdata=rdata, type=response_type))


def mgcv_posterior_samples(gam, data, n_samples=100, response_type='inverse_link'):

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
        if family == 'gaussian' and link == 'identity':
            samples = _post
        elif family == 'binomial' and link == 'logit':
            samples = 1. / (1. + np.exp(-_post))
        elif family == 'poisson' and link == 'log':
            samples = np.exp(-_post)
        else:
            raise NotImplementedError

    elif response_type == 'response':
        family, link = get_family(gam), get_link(gam)
        if family == 'gaussian' and link == 'identity':
            sigma_noise = np.sqrt(gam[19]) #TODO check
            samples = _post + np.random.normal(0, sigma_noise, _post.size).reshape(_post.shape)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return samples


def get_family(gam):
    return gam[3][0][0]

def get_link(gam):
    return gam[3][1][0]

def get_names(obj):
    return rbase.names(obj)

def summary(model):
    print(rbase.summary(model))

