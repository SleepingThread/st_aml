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


Main principles
===============

st_aml.blocks
-------------

.. note::
    I think one no needs to implement multiple 
    <manipulation>_block methods - the same result
    can be obtained by putting multiple data manipulations
    into process_block and then choose one of theese 
    manipulations though method parameters.

.. note::
    Block.process_block ( data , \**params ) method 
    must process only specific keys inside DataBlock's.
    Other keys must be copied without changing.


* BaseNetwork and Block classes support two types of wotk:
    * training mode ( engine ) : fit = True
    * predict mode ( core ) : fit = False

* Block class:
    * can be dynamic and static 
      (without defined n_outputs or not)
    * needs to store data manipulation algorithms itself
    * implements process(data, manipulations, <fit_mode>)
      method to manage data manipulation
      methods. For example: 
      * fit_block(data, \**params)
      * process_block(data, \**params)

    * one can implement manipulation method - 
      <manipulation>_block (transform_block for example)

    * others can implement different manipulations 
      inside process_block method and control
      which method to use through process_block \**params.

    * process ( data, manipulations, fit ) method
      implements method to execute manipulations.

    * process_tensors ( tensors , manipulations ) method
      provides a way to implement block using keras library.
      This method must produce equivalent to the process
      method manipulations.

* _NetworkNode class:
    * process ( network, fit ) , 
      process_tensors ( network, tens_dict ) - methods 
      to execute underlying block 
      process and process_tensors methods.



Inheritance diagrams
====================

st_aml.blocks
-------------

.. inheritance-diagram:: st_aml.blocks


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
