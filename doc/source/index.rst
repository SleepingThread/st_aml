.. st_aml documentation master file, created by
   sphinx-quickstart on Mon Feb 25 10:14:47 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to st_aml's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

About st_aml module
==================================

.. code-block:: python
    :linenos:

    ds = SDFDataset(sdf_file="bzr.sdf",target_prefix="IC50_uM",target=st_aml.targets.IntervalTarget())
    cdk_descr = st_qsar.descriptors.CDKDescriptors()
    # hack to prevent errors
    cdk_descr.calculate([ds.objects[0]])
    description = st_aml.descriptors.Description([cdk_descr])
    # descr - numpy.ndarray with 2 dimensions
    descr = description.calculate(ds.objects,verbose=1)
    mat,description2 = description.remove_nan_descriptors(descr)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
