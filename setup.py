# !/usr/bin/env python



from distutils.core import setup
try:  # pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # pip <= 9.0.3
    from pip.req import parse_requirements


# parse_requirements() returns generator of pip.req.InstallRequirement objects
reqs = parse_requirements(resource('requirements.txt'), session=False)
reqs = [str(ir.req) for ir in reqs]

setup(
    name='hciplot',
    packages=['hciplot'],
    version='0.1.0',
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
        'Topic :: Software Development',
    ],
)
