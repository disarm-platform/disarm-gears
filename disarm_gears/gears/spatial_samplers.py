import numpy as np
from GPyOpt.methods import BayesianOptimization


class SpatialSampler:

    def __init__(self, f, domain, maximize=False):
        '''General class for samplers.'''
        assert callable(f), 'Object is not callable.'
        assert isinstance(maximize, bool), 'Expecting True/False value.'
        self.maximize = maximize
        self.f = -f if self.maximize else f
        self.domain = domain#TODO assert domain

    def _domain_grid(self):
        '''Return a grid of the spatial domain.'''
        _grids = [_d['domain'] if _d['type'] == 'discrete' else
                  np.linspace(*_d['domain']) for _d in self.domain]
        return np.vstack([_m.ravel() for _m in np.meshgrid(*_grids)]).T

    def choose_location(self):
        raise NotImplementedError


class RandomSampler(SpatialSampler):

    def __init__(self, f, domain):
        '''Random sampler.'''
        super(RandomSampler, self).__init__(f=f, domain=domain, maximize=False)

    def choose_location(self, grid=None, **kwargs):
        '''Choose a location across domain to take a sample.'''
        #TODO handle mixed domains
        if grid is None:
            _s = [np.random.permutation(_d['domain'])[0] if _d['type'] == 'discrete' \
                      else np.random.uniform(*_d['domain']) for _d in self.domain]
        else:
            _s = np.random.permutation(grid)[0]
        self.X_optim = _s[0] if len(_s) == 1 else np.array([_s])


class BruteForce(SpatialSampler):

    def __init__(self, f, domain, maximize=False):
        super(RandomSampler, self).__init__(f=f, domain=domain, maximize=maximize)

    def choose_location(self, grid=None, **kwargs):
        '''Choose a location across domain to take a sample.'''
        if grid is None:
            X = self._domain_grid()
        else:
            #TODO Assert grid and domain match
            X = grid

        _ix = np.argmin([self.f(x[None, :]) for x in X])
        self.X_optim = X[_ix] if X.shape[1] == 1 else X[_ix].reshape(1, -1)


class GPyBOpt(SpatialSampler):

    def __init__(self, **kwargs):
        '''This is a wrapper of the GPyOpt functionality.'''
        super(GPyBOpt, self).__init__(f=f, domain=domain, maximize=maximize)
        self.model = BayesianOptimization(**kwargs)

    def choose_location(self, **kwargs):
        '''Choose a location across domain to take a sample.'''
        self.model.run_optimization(**kwargs)
        self.X_optim = self.model.x_opt
        self.X_optim = self.X_optim.ravel() if self.X_optim.size == 1 else self.X_optim.reshape(1, -1)
