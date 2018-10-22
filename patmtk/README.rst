Green Magic - Python Library
================================

Green Magic is a library containing class models allowing users to train machine learning models as well as visualize cannabis strain data. It has functionality for encoding raw cannabis strain data into features usefull for visualization and cluster analysis. It contains implementations for model evaluation and methods for data exploration.

Key features of the Library:

* Data cleaning
* Seemless dataset creation
* Extendable feature extraction system
* Usage of the Somoclu library [1] as the backend, which allows for 'fast execution of Self-Organizing Maps by parallelization: OpenMP and CUDA are supported'.
* Visualization of maps
* Kmeans and Affinity-propagation based clustering
* Formatted print of statistics and distributions


Usage
-----
A simple example is below.

::

    from green_magic import WeedMaster
    from green_magic.clustering import ClusteringFactory, DistroReporter, get_model_quality_reporter
    all_vars = ['type', 'effects', 'medical', 'negatives', 'flavors']
    active_vars = ['type', 'effects', 'medical', 'negatives', 'flavors']
    wd = 'pd'
    wm = WeedMaster()
    dt = wm.create_weedataset(dt_path, wd)
    dt.use_variables(active_vars)
    dt.clean()
    vectors = wm.get_feature_vectors(dt)
    print(dt)
    wm.save_dataset(wd)
    som = wm.map_manager.get_som('toroid.rectangular.30.30.pca')
    wm.map_manager.show_mmap(som)
    clf = ClusteringFactory(wm)
    cls = clf.create_clusters(som, 'kmeans', nb_clusters=10, vars=all_vars, ngrams=1)
    print(cls)
    cls.print_map()
    r = DistroReporter()
    r.print_distros(cls0, 'type', prec=3)
    qr = get_model_quality_reporter(wm, wd)
    print(qr.measure(cls, metric='silhouette'))
    print(qr.measure(cls, metric='cali-hara'))


Installation
------------
The code is available on PyPI, hence it can be installed by

::

    $ pip install green_magic

Citation
--------

1. Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2017). Somoclu: An Efficient Parallel Library for Self-Organizing Maps.  Journal of Statistical Software, 78(9), pp.1--21. DOI:`10.18637/jss.v078.i09 <https://doi.org/10.18637/jss.v078.i09>`_. arXiv:`1305.1422 <https://arxiv.org/abs/1305.1422>`_.
