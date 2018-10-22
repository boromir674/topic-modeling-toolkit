from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='green_magic',
    version='0.5.7',
    description='The Green Magic library of the Green-Machine',
    long_description=readme(),
    keywords='cannabis strain self-organizing maps visualization',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'Intended Audience :: Science/Research',
        ],
    url='https://github.com/boromir674/green-magic',
    author='Konstantinos',
    author_email='k.lampridis@hotmail.com',
    license='GNU GPLv3',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=['numpy', 'nltk', 'sklearn', 'somoclu', 'pandas'],
    include_package_data=True,
    test_suite='green_magic.tests',
    zip_safe=False
)
