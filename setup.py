# !/usr/bin/env python

import os
import re
from setuptools import setup
try:  # pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # pip <= 9.0.3
    from pip.req import parse_requirements


def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)


with open(resource('hciplot', '__init__.py')) as version_file:
    version_file = version_file.read()
    VERSION = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                        version_file, re.M)
    VERSION = VERSION.group(1)


# parse_requirements() returns generator of pip.req.InstallRequirement objects
reqs = parse_requirements(resource('requirements.txt'), session=False)
reqs = [str(ir.req) for ir in reqs]

setup(
    name='hciplot',
    packages=['hciplot'],
    version=VERSION,
    description='High-contrast Imaging Plotting library',
    author='Carlos Alberto Gomez Gonzalez',
    license='MIT',
    author_email='carlosgg33@gmail.com',
    url='https://github.com/carlgogo/hciplot',
    keywords=['plotting', 'hci', 'package'],
    install_requires=reqs,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development'],
)
