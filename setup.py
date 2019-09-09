from setuptools import setup, find_packages

from os import path

this_dir = path.dirname(path.realpath(__file__))

def readme():
    with open(path.join(this_dir, 'README.rst')) as f:
        return f.read()


setup_kwargs = dict(
    name='topic-modeling-toolkit',
    version='0.5.6',
    description='Topic Modeling Toolkit',
    long_description=readme(),
    keywords='topic modeling machine learning',

    # project_urls={
    #     "Source Code": "https://github.com/..,
    # },
    zip_safe=False,

    # what packages/distributions (python) need to be installed when this one is. (Roughly what is imported in source code)
    install_requires=[
        'numpy', 'scipy', 'EasyPlot==1.0.0', 'nltk',
        'pandas', 'gensim', 'tqdm', 'in-place', 'protobuf',
        'click', 'future', 'attrs',
        'PyInquirer',  # # for the transform.py interface
        # 'configparser'  # to make statement 'from configparser import ConfigParser' python 2 and 3 compatible
    ],
    # A string or list of strings specifying what other distributions need to be present in order for the setup script to run.
    # (Note: projects listed in setup_requires will NOT be automatically installed on the system where the setup script is being run.
    # They are simply downloaded to the ./.eggs directory if they're not locally available already. If you want them to be installed,
    # as well as being available when the setup script is run, you should add them to install_requires and setup_requires.)
    # setup_requires=[],

    # Folder where unittest.TestCase-like written modules reside. Specifying this argument enables use of the test command
    # to run the specified test suite, e.g. via setup.py test.
    test_suite='tests',

    # Declare packages that the project's tests need besides those needed to install it. A string or list of strings specifying
    # what other distributions need to be present for the package's tests to run. Note that these required projects will not be installed on the system where the
    # tests are run, but only downloaded to the project's setup directory if they're not already installed locally.
    # Use to ensure that a package is available when the test command is run.
    tests_require=['pytest'],

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Science/Research'
    ],
    url='https://github.com/boromir674/topic-modeling-toolkit',
    # download_url='point to tar.gz',  # help easy_install do its tricks
    author='Konstantinos Lampridis',
    author_email='k.lampridis@hotmail.com',
    license='GNU GPLv3',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # this is required by distutils
    # py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    # Include all data files in packages that distutils are aware of through the MANIFEST.in file
    # package_data={
    #     # If any package contains *.txt or *.rst files, include them:
    #     '': ['*.txt', '*.rst'],
    #     'package_name.file_name': ['data/*.txt', 'data/model.pickle'],
    # },
    entry_points={
        'console_scripts': [
            'transform = topic_modeling_toolkit.transform:main',
            'train = topic_modeling_toolkit.train:main',
            'tune = topic_modeling_toolkit.tune:main',
            'make-graphs = topic_modeling_toolkit.make_graphs:main',
            'report-datasets = topic_modeling_toolkit.report_datasets:main',
            'report-models = topic_modeling_toolkit.report_models:main',
            'report-topics = topic_modeling_toolkit.report_topics:main',
            'report-kl= topic_modeling_toolkit.report_kl:main',
        ]
    },
    # A dictionary mapping names of "extras" (optional features of your project: eg imports that a console_script uses) to strings or lists of strings
    # specifying what other distributions must be installed to support those features.
    # extras_require={},

)

setup(**setup_kwargs)
