from distutils.core import setup

__version__ = '0.1'

setup(name = 'DisarmGears',
      version = __version__,
      description = "Disarm's Brains on Demand Toolbox",
      author = 'DiSARM authors',
      author_email = 'ric70x7@gmail.com',
      url = 'www.disarm.io',
      packages = ['disarm_gears',],
      license = open('LICENSE').read(),
      long_description = open('README').read(),
      )
