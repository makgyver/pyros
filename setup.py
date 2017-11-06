from setuptools import setup
#from distutils.core import setup

setup(
  name = 'mkpyros',
  packages = ['pyros'],
  version = '0.9.10',
  install_requires=[
        "numpy",
        "scipy",
        "cvxopt"
  ],
  license = "MIT",
  description = 'Python module for building and evaluating recommender systems for implicit feedback.',
  author = 'Mirko Polato',
  author_email = 'mak1788@gmail.com',
  url = 'https://github.com/makgyver/pyros',
  download_url = 'https://github.com/makgyver/pyros',
  keywords = ['recommendation-algorithm', 'collaborative-filtering', 'algorithm', 'kernel'],
  classifiers = [
                 'Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'License :: OSI Approved :: MIT License',
                ]
)