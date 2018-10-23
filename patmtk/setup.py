from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='patmtk',
    version='0.5',
    description='Perspectivew Aware Topic Modeling Toolkit',
    long_description=readme(),
    keywords='topic modeling machine learning',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Science/Research',
        ],
    # url='https://github.com/boromir674/....',
    author='Konstantinos',
    author_email='k.lampridis@hotmail.com',
    license='GNU GPLv3',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=['numpy', 'nltk'],
    include_package_data=True,
    test_suite='patm.tests',
    zip_safe=False
)
