from distutils.core import setup

__version__ = '0.1'

setup(name = 'DisarmGears',
      version = __version__,
      description = "Disarm's Brains on Demand Toolbox",
      author = 'DiSARM authors',
      url = 'www.disarm.io',
      packages = ['DisarmGears',],
      license = 'MIT License',
      long_description = open('README.md').read(),
      )
