''' 
               setup.py

          author : pearcandy
          date   : 2022/4/22               
                                           '''

#coding:utf-8

from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="phbin",
    version="0.0.2",
    license="GNU lv3",
    description=
    "A simple tool of persistent homology analysis for binary images",
    long_description="README.rst",
    author="Yasutaka Nishida",
    url="https://github.com/pearcandy/phbin",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={'': ['utils/*', 'hc/*', 'ml/*']},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt'),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    python_requires='>=3.8',
)
