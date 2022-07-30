#!/usr/bin/env python



import sys
import logging
from setuptools import setup, find_packages


# Set up the logging environment
logging.basicConfig()
log = logging.getLogger()

# Handle the -W all flag
if 'all' in sys.warnoptions:
    log.level = logging.DEBUG

# Get version from the module
with open('ArtificialVocalLearning/__init__.py') as f:
    for line in f:
        if line.find('__version__') >= 0:
            version = line.split('=')[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

# Dependencies
DEPENDENCIES = [
    'numpy',
    'pyMetaheuristic',
    'tensorflow>=2.8'
    'tools_io>=0.1',
    'VocalTractLab==0.4.23',
]

CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU General Public License v3 (GPLv3)
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Software Development
Topic :: Scientific/Engineering
Typing :: Typed
Operating System :: Microsoft :: Windows
Operating System :: Unix
"""

setup_args = dict(
    name='artificial-vocal-learning',
    version=version,
    description='A high quality vocal learning simulation for Python',
    #long_description= DOCLINES,
    url='https://github.com/paul-krug/artificial-vocal-learning',
    #download_url=,
    author='Paul Krug',
    author_email='paul_konstantin.krug@tu-dresden.de',
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    keywords=[ 'vocal learning', 'simulation', 'Python' ],
    packages=find_packages(),
    package_dir={'artificial-vocal-learning': 'ArtificialVocalLearning'},
    install_requires=DEPENDENCIES,
    zip_safe= True,
)

setup(**setup_args)