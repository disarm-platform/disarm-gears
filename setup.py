from distutils.core import setup

__version__ = '.0.1'

setup(name = 'DisarmGears',
      version = __version__,
      description = "(Disarm's Brains on Demand Toolbox)",
      packages = ['DisarmGears',],
      license = 'BSD 3-clause',
      long_description=open('README.md').read(),
      )
