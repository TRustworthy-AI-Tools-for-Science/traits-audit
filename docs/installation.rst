Installation
============

Requirements: Python ≥ 3.11.  Core dependencies are ``numpy``, ``scipy``,
``matplotlib``, and ``kaleido``.  All optional extras are listed below.


uv workspace (recommended)
---------------------------

If ``traits-audit`` is a member of your uv workspace, a single command
installs the package in editable mode *and* all four demo extras:

.. code-block:: bash

   uv sync

Source changes take effect immediately without reinstalling.  Works on
Linux, macOS, and Windows.


Standalone pip install
----------------------

.. code-block:: bash

   pip install "."             # core only
   pip install ".[mlflow]"     # + MLflow logging
   pip install ".[pybamm]"     # + PyBAMM demo
   pip install ".[sdl]"        # + self-driving lab demo
   pip install ".[camd]"       # + materials screening demo


Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 30

   * - Extra
     - Key packages
     - Required for
   * - ``mlflow``
     - mlflow, plotly, matplotlib
     - MLflow experiment tracking and figure logging
   * - ``pybamm``
     - pybamm, scikit-learn
     - Li-ion C-rate optimisation demo (``ta-pybamm-demo``)
   * - ``sdl``
     - self-driving-lab-demo, ax-platform
     - Self-driving lab LED demo (``ta-sdl-demo``)
   * - ``camd``
     - scikit-learn, pandas, pymatgen, matminer, django, PuLP
     - Materials stability screening demo (``ta-camd-demo``)
   * - ``docs``
     - sphinx, furo, myst-parser, sphinx-autodoc-typehints
     - Building this documentation
   * - ``dev``
     - pytest, pytest-cov
     - Running the test suite

.. note::

   ``ta-camd-demo`` uses a scikit-learn BaggingRegressor surrogate and does
   not require the ``camd`` Python package.  On first run it attempts to
   download the OQMD Voronoi-Magpie fingerprints dataset (~150 MB) from
   `data.matr.io <https://data.matr.io>`_ and caches it under
   ``~/.cache/traits_audit/``.  If the download fails, synthetic data with
   the same schema is used automatically.


Conda environment
-----------------

Use the conda env as the base and install pip packages on top:

.. code-block:: bash

   conda activate traits-audit
   pip install -e "."
   pip install -e ".[mlflow,camd,pybamm,sdl]"


Verifying the install
---------------------

.. code-block:: python

   from traits_audit import AuditHook, AuditPipeline
   from traits_audit.checks import CalibrationErrorCheck
   print("OK")
