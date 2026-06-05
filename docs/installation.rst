Installation
============

Requirements: Python ≥ 3.12.  Core dependencies are ``numpy`` and
``scipy``.  ``mlflow`` is optional.


uv workspace (recommended)
---------------------------

If ``traits_audit`` is a member of your uv workspace, run:

.. code-block:: bash

   uv sync --all-packages

This installs the package in editable mode.  Changes to the source take
effect immediately without reinstalling.


Standalone install
------------------

.. code-block:: bash

   pip install ./traits_audit


Conda environment
-----------------

.. code-block:: bash

   pip install -e ./traits_audit          # editable


Verifying the install
---------------------

.. code-block:: python

   from traits_audit import AuditHook, AuditPipeline
   from traits_audit.checks import CalibrationErrorCheck
   print("OK")
