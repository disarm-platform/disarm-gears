from distutils.core import setup

__version__ = "0.1.4"

setup(name = "DisarmGears",
      version = __version__,
      description = "Disarm's Brains on Demand Toolbox",
      author = "DiSARM authors",
      author_email = "ric70x7@gmail.com",
      url = "www.disarm.io",
      packages = ["disarm_gears",
                  "disarm_gears.chain_drives",
                  "disarm_gears.chain_drives.prototypes",
                  "disarm_gears.frames",
                  "disarm_gears.util",
                  "disarm_gears.testing",
                  "disarm_gears.validators"],
      license = open("LICENSE").read(),
      long_description = open("README.md").read(),
      )
