from setuptools import setup, find_packages
#from distutils.core import setup

setup(
  name = 'mkpyros',
  packages = find_packages(exclude=['build', '_docs', 'templates']),
  version = '0.9.20',
  install_requires=[
        "numpy",
        "scipy",
        "cvxopt"
  ],
  license = "MIT",
  description = 'Python module for building and evaluating recommender systems for implicit feedback. Full documentation @ http://mkpyros.readthedocs.io/en/latest/',
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