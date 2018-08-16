import numpy as np
from disarm_gears.validators import validate_2d_array
from GPyOpt.methods import BayesianOptimization


class SpatialSampler:

    def __init__(self, f, domain, maximize=False):
        '''General class for samplers.

        :param f: Function used as criterion for sample selection
                  (i.e., the location chosen is the one that optimizes f).
                  A callable object.
        :param domain: Definition of the space where f can be evaluated.
                       List of two dictionaries with the following structure:
                       [{'name': 'lng', 'type': 'continuous', 'domain': (x1, x2)},
                        {'name': 'lat', 'type': 'continuous', 'domain': (y1, y2)}
        :param maximize: Whether the optimization of the criterion f requires maximization.
                         False (default) implies minimization of f.
                         Bool object.
        '''
        assert callable(f), 'Object is not callable.'
        assert isinstance(maximize, bool), 'Expecting True/False value.'
        self.maximize = maximize
        self.f = f if not self.maximize else lambda x: -f(x)

        self._validate_domain(domain)
        self.domain = domain

    def _validate_domain(self, domain):
        '''Validate that domain is adequately defined.'''
        # Check list is correct
        assert isinstance(domain, list), 'Object is expected to be a list.'
        assert len(domain) == 2, 'Length of domain is expected to be 2.'
        assert isinstance(domain[0], dict) and isinstance(domain[0], dict),\
            'Object is expected to be a list with two dictionaries. '

        # Check fields required are contained in each dictionary
        _fields = ['name', 'type', 'domain']
        assert np.all([[fi in di for fi in _fields] for di in domain]), 'Incomplete domain definition.'

        for di in domain:
            if di['type'] == 'continuous':
                assert len(di['domain']) == 2, 'Range of domain not understood.'
                assert di['domain'][0] < di['domain'][1], 'Range of domain not understood.'
            elif di['type'] == 'discrete':
                assert isinstance(di['domain'], np.ndarray), 'Expecting a numpy array.'
                assert di['domain'].size == np.unique(di['domain']).size,\
                    'Domain has repeated categories.'
            else:
                raise NotImplementedError

    def _validate_set(self, X):
        '''
        Verify that a set of points X belongs to the domain .

        :param X: Set of spatial points.
                  Numpy array, shape [n, 2]
        '''
        validate_2d_array(X, n_cols=2)
        for j in range(2):
            if self.domain[j]['type'] == 'continuous':
                a, b = self.domain[j]['domain']
                assert np.all(np.logical_and(a <= X[:, j], X[:, j] <= b)), 'X not in domain.'
            elif self.domain[j]['type'] == 'discrete':
                assert np.all([xi in self.domain[j]['domain'] for xi in X[:, j]]), 'X not in domain.'
            else:
                raise NotImplementedError

    def _domain_grid(self, n_continuous=50):
        '''
        Return a grid of the spatial domain.

        :param n_continuous: Number of grid points for continuous domains.
                             Positive integer
        '''
        assert n_continuous > 0
        assert round(n_continuous) == n_continuous
        _grids = [_d['domain'] if _d['type'] == 'discrete' else
                  np.linspace(*_d['domain'], n_continuous) for _d in self.domain]
        return np.vstack([_m.ravel() for _m in np.meshgrid(*_grids)]).T

    def choose_location(self):
        raise NotImplementedError


class RandomSampler(SpatialSampler):

    def __init__(self, f, domain=None):
        '''Random sampler.'''
        super(RandomSampler, self).__init__(f=f, domain=domain, maximize=False)

    def choose_location(self, X=None, **kwargs):
        '''
        Choose a location across domain to take a sample.

        :param X: Domain subset where the location search is constrained.
                  Numpy array, shape [n, 2].
        '''
        if X is None:
            _s = [np.random.permutation(_d['domain'])[0] if _d['type'] == 'discrete' \
                      else np.random.uniform(*_d['domain']) for _d in self.domain]
        else:
            self._validate_set(X)
            _s = np.random.permutation(X)[0]
        self.X_optim = _s[0] if len(_s) == 1 else np.array([_s])


class GridSearchSampler(SpatialSampler):

    def __init__(self, f, domain, maximize=False):
        '''
        Sample selector based on a grid search.
        '''
        super(GridSearchSampler, self).__init__(f=f, domain=domain, maximize=maximize)

    def choose_location(self, X=None, **kwargs):
        '''
        Choose a location across domain to take a sample.

        :param X: Domain subset where the location search is constrained.
                  Numpy array, shape [n, 2].
        '''
        if X is None:
            X = self._domain_grid()
        else:
            self._validate_set(X)
        _ix = np.argmin(self.f(X)) #[self.f(x[None, :]) for x in X])
        self.X_optim = X[slice(_ix, _ix + 1)]


class GPyOptSampler(SpatialSampler):

    def __init__(self, **kwargs):
        '''This is a wrapper of the GPyOpt functionality.'''
        assert 'f' in kwargs.keys(), 'Missing argument f.'
        assert 'domain' in kwargs.keys(), 'Missing argument domain.'
        f = kwargs['f']
        domain = kwargs['domain']
        maximize = False if 'maximize' not in kwargs.keys() else kwargs['maximize']
        super(GPyOptSampler, self).__init__(f=f, domain=domain, maximize=maximize)
        self.model = BayesianOptimization(**kwargs)

    def choose_location(self, **kwargs):
        '''Choose a location across domain to take a sample.'''
        self.model.run_optimization(**kwargs)
        self.X_optim = self.model.x_opt
        self.X_optim = self.X_optim.reshape(1, -1)
