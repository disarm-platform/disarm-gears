from distutils.core import setup

__version__ = "0.2.0"

setup(name="DisarmGears",
      version=__version__,
      description="Disarm's Brains on Demand Toolbox",
      author="DiSARM authors",
      author_email="ric70x7@gmail.com",
      url="www.disarm.io",
      packages=["disarm_gears",
                "disarm_gears.chain_drives",
                  "disarm_gears.chain_drives.prototypes",
                  "disarm_gears.frames",
                  "disarm_gears.util",
                  "disarm_gears.testing",
                  "disarm_gears.r_plugins",
                  "disarm_gears.validators"],
      install_requires=["numpy", "scipy", "scikit-learn", "pandas >= 0.22.0", "geopandas",
                        "shapely", "descartes", "matplotlib", "rtree", "pygam", "requests", "rpy2"],
      license=open("LICENSE").read(),
      long_description=open("README.md").read(),
      )
