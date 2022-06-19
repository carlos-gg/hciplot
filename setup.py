# !/usr/bin/env python

import os
import re
from setuptools import setup
try:
    # pip >=20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements

def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)

# parse_requirements() returns generator of pip.req.InstallRequirement objects
reqs = parse_requirements(resource('requirements.txt'), session=PipSession)
requirements = [str(ir.requirement) for ir in reqs]


with open(resource('hciplot', '__init__.py')) as version_file:
    version_file = version_file.read()
    VERSION = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                        version_file, re.M)
    VERSION = VERSION.group(1)

with open(resource('README.md')) as readme_file:
    README = readme_file.read()

setup(
    name='hciplot',
    packages=['hciplot'],
    version=VERSION,
    description='High-contrast Imaging Plotting library',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Carlos Alberto Gomez Gonzalez, Valentin Christiaens',
    license='MIT',
    author_email='valentinchrist@hotmail.com',
    url='https://github.com/carlgogo/hciplot',
    download_url='https://github.com/carlgogo/hciplot/archive/refs/tags/v0.1.8.tar.gz',
    keywords=['plotting', 'hci', 'package'],
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Astronomy'],
)
