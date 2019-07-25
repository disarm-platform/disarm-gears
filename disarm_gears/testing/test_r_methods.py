import pytest
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity
from disarm_gears.r_plugins import r_methods
from rpy2.robjects.packages import importr
rstats = importr('stats')
rbase = importr('base')

np.random.seed(10)

## Blobs and grid
X = make_blobs(n_samples=120, n_features=2, centers=10, cluster_std=.6)[0]
f_size = 15
g_size = 50
f1, f2 = np.meshgrid(np.linspace(-9.6, 9.6, f_size), np.linspace(-9.6, 9.6, f_size))
g1, g2 = np.meshgrid(np.linspace(-10, 10, g_size), np.linspace(-10, 10, g_size))


## Density
bw = 2
kde = KernelDensity(bandwidth=bw, kernel='gaussian', algorithm='ball_tree')
kde.fit(X)
hf = np.exp(kde.score_samples(np.vstack([f1.flatten(), f2.flatten()]).T))
hg = np.exp(kde.score_samples(np.vstack([g1.flatten(), g2.flatten()]).T))
a = hg.max()
hg = hg / a
hf = hf / a

x1 = np.random.normal(0, .1, hf.size)
x2 = np.random.normal(0, .1, hf.size)
y = .5 * x1 + .2 * x2 + hf.ravel()


f_data = pd.DataFrame({'lng': f1.ravel(), 'lat': f2.ravel(), 'x1': x1, 'x2': x2, 'y': y})
f_data2 = f_data.copy()
f_data2.loc[:, 'y'] = np.round(100 * 1. / (1. + np.exp(-f_data2.y)))
f_data2['n'] = 100

f_data3 = f_data.copy()
f_data3.loc[:, 'lng'] = f_data.lng / 20
f_data3.loc[:, 'lat'] = f_data.lat / 20


def test_pdframe2rdframe():

    with pytest.raises(AssertionError):
        r_methods.pdframe2rdframe(np.array(f_data))

    rdframe = r_methods.pdframe2rdframe(f_data)
    assert rdframe.ncol == 5
    assert np.all([ci in ['lng', 'lat', 'x1', 'x2', 'y'] for ci in rdframe.colnames])
    assert rdframe.nrow == 225
    assert isinstance(rdframe, robjects.vectors.DataFrame)


def test_mgcv_fit():

    formula_gauss = "y ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    formula_binom = "cbind(y, 100 - y) ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    formula_poiss = "y ~ offset(log(n)) + x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    weights = np.random.uniform(.1, 1, f_data2.shape[0])

    m_binom = r_methods.mgcv_fit(formula_binom, f_data2, family='binomial', weights=None, method='REML')
    m_poiss = r_methods.mgcv_fit(formula_poiss, f_data2, family='poisson', weights=None, method='REML')
    m_gauss = r_methods.mgcv_fit(formula_gauss, f_data2, family='gaussian', weights=weights, method='REML')

    m_gauss_bam = r_methods.mgcv_fit(formula_gauss, f_data2, family='gaussian', weights=None, method='fREML',
                                     bam=True)

    assert isinstance(m_binom, robjects.vectors.ListVector)
    assert isinstance(m_poiss, robjects.vectors.ListVector)
    assert isinstance(m_gauss, robjects.vectors.ListVector)
    assert isinstance(m_gauss_bam, robjects.vectors.ListVector)

    with pytest.raises(NotImplementedError):
        r_methods.mgcv_fit(formula_binom, f_data2, family='xx', weights=None, method='REML')

    with pytest.raises(NotImplementedError):
        r_methods.mgcv_fit(formula_binom, f_data2, family='xx', weights=weights[:40], method='REML')
    with pytest.raises(NotImplementedError):
        r_methods.mgcv_fit(formula_binom, f_data2, family='xx', weights=weights[:, None], method='REML')
    with pytest.raises(NotImplementedError):
        r_methods.mgcv_fit(formula_binom, f_data2, family='xx', weights=list(weights), method='REML')
    with pytest.raises(NotImplementedError):
        r_methods.mgcv_fit(formula_gauss, f_data2, family='gaussian', weights=weights, method='fREML', bam=True)


def test_mgcv_predict():

    formula_binom = "cbind(y, 100 - y) ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    m_binom = r_methods.mgcv_fit(formula_binom, f_data2, family='binomial', weights=None, method='REML')
    response = r_methods.mgcv_predict(m_binom, f_data2, response_type='response')
    link = r_methods.mgcv_predict(m_binom, f_data2, response_type='link')
    lpmatrix = r_methods.mgcv_predict(m_binom, f_data2, response_type='lpmatrix')

    assert isinstance(response, np.ndarray)
    assert isinstance(link, np.ndarray)
    assert isinstance(lpmatrix, np.ndarray)
    assert response.size == f_data2.shape[0]
    assert link.size == f_data2.shape[0]
    assert lpmatrix.shape[0] == f_data2.shape[0]
    assert lpmatrix.shape[1] == len(rstats.coef(m_binom))

    with pytest.raises(NotImplementedError):
        r_methods.mgcv_predict(m_binom, f_data2, response_type='xx')


def test_mgcv_posterior_samples():

    formula_gauss = "y ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    formula_binom = "cbind(y, 100 - y) ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    formula_poiss = "y ~ offset(log(n)) + x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    m_gauss = r_methods.mgcv_fit(formula_gauss, f_data2, family='gaussian', weights=None, method='REML')
    m_binom = r_methods.mgcv_fit(formula_binom, f_data2, family='binomial', weights=None, method='REML')
    m_poiss = r_methods.mgcv_fit(formula_poiss, f_data2, family='poisson', weights=None, method='REML')
    link = r_methods.mgcv_posterior_samples(m_binom, f_data2, n_samples=100, response_type='link')
    ilink_gauss = r_methods.mgcv_posterior_samples(m_gauss, f_data2, n_samples=100, response_type='inverse_link')
    ilink_binom = r_methods.mgcv_posterior_samples(m_binom, f_data2, n_samples=100, response_type='inverse_link')
    ilink_poiss = r_methods.mgcv_posterior_samples(m_poiss, f_data2, n_samples=100, response_type='inverse_link')
    resp_gauss = r_methods.mgcv_posterior_samples(m_gauss, f_data2, n_samples=100, response_type='response')

    assert isinstance(link, np.ndarray)
    assert link.shape[0] == 100
    assert link.shape[1] == f_data2.shape[0]

    assert isinstance(ilink_gauss, np.ndarray)
    assert isinstance(ilink_binom, np.ndarray)
    assert isinstance(ilink_poiss, np.ndarray)

    assert isinstance(resp_gauss, np.ndarray)

    with pytest.raises(NotImplementedError):
        r_methods.mgcv_posterior_samples(m_binom, f_data2, n_samples=100, response_type='xx')

    with pytest.raises(NotImplementedError):
        r_methods.mgcv_posterior_samples(m_binom, f_data2, n_samples=100, response_type='response')


def test_mgcv_get_rho_power_exp2():

    formula_gauss = "y ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2%s)), d=2, k=-1)"

    with pytest.raises(NotImplementedError):
        r_methods.mgcv_get_rho_power_exp2(iter_formula=formula_gauss, data=f_data, smooth_dim=1, family='gaussian',
                                          weights=None, method="REML")
    with pytest.raises(NotImplementedError):
        r_methods.mgcv_get_rho_power_exp2(iter_formula=formula_gauss, data=f_data, smooth_dim=3, family='gaussian',
                                          weights=None, method="REML")

    rho_gauss = r_methods.mgcv_get_rho_power_exp2(iter_formula=formula_gauss, data=f_data, smooth_dim=2, family='gaussian',
                                                  weights=None, method="REML")
    rho_gauss3 = r_methods.mgcv_get_rho_power_exp2(iter_formula=formula_gauss, data=f_data3, smooth_dim=2, family='gaussian',
                                                  weights=None, method="REML")

    assert isinstance(rho_gauss, float)
    assert isinstance(rho_gauss3, float)


def test_get_family():

    formula_binom = "cbind(y, 100 - y) ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    m_binom = r_methods.mgcv_fit(formula_binom, f_data2, family='binomial', weights=None, method='REML')
    family = r_methods.get_family(m_binom)
    assert family == 'binomial'


def test_get_link():

    formula_binom = "cbind(y, 100 - y) ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    m_binom = r_methods.mgcv_fit(formula_binom, f_data2, family='binomial', weights=None, method='REML')
    link = r_methods.get_link(m_binom)
    assert link == 'logit'


def test_get_names():

    rdframe = r_methods.pdframe2rdframe(f_data)
    obj_names1 = r_methods.get_names(rdframe)
    assert isinstance(obj_names1, list)

    formula_binom = "cbind(y, 100 - y) ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    m_binom = r_methods.mgcv_fit(formula_binom, f_data2, family='binomial', weights=None, method='REML')
    obj_names2 = r_methods.get_names(m_binom)
    assert isinstance(obj_names2, list)


def test_summary(capfd):

    formula_binom = "cbind(y, 100 - y) ~ x1 + x2 + te(lng, lat, bs='gp', m=list(c(2, 4, 2)), d=2, k=-1)"
    m_binom = r_methods.mgcv_fit(formula_binom, f_data2, family='binomial', weights=None, method='REML')
    r_methods.summary(m_binom)
    out, err = capfd.readouterr()
    assert out is not None
