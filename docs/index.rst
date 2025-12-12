Neural DSL Documentation
========================

Welcome to Neural DSL's documentation!

Neural DSL is a domain-specific language and debugger for neural networks that provides
a declarative syntax for defining, training, debugging, and deploying neural networks
with cross-framework support (TensorFlow, PyTorch, ONNX).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   dsl
   cli
   api/index

Features
--------

* **Declarative DSL**: Define neural networks using a clean, intuitive syntax
* **Multi-Backend**: Generate code for TensorFlow, PyTorch, and ONNX
* **Shape Propagation**: Automatic tensor shape inference and validation
* **Real-time Debugging**: Interactive dashboard for monitoring training
* **HPO Support**: Built-in hyperparameter optimization with Optuna
* **Cloud Integration**: Deploy to Kaggle, Colab, and AWS SageMaker

Installation
------------

.. code-block:: bash

   pip install neural-dsl
   pip install neural-dsl[full]  # with all optional dependencies

Quick Start
-----------

.. code-block:: python

   # Define a model in Neural DSL
   model = """
   Network MNIST_Classifier {
       Input: shape=(1, 28, 28)
       
       Conv2D: filters=32, kernel_size=3, activation=ReLU
       MaxPooling2D: pool_size=2
       Conv2D: filters=64, kernel_size=3, activation=ReLU
       MaxPooling2D: pool_size=2
       Flatten
       Dense: units=128, activation=ReLU
       Output: units=10, activation=Softmax
       
       Optimizer: Adam(learning_rate=0.001)
       Loss: CrossEntropy
   }
   """

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
