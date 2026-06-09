Installation
============

Requirements: Python ≥ 3.11.  Core dependencies are ``numpy``, ``scipy``,
``matplotlib``, and ``kaleido``.  All optional extras are listed below.


uv workspace (recommended)
---------------------------

If ``traits-audit`` is a member of your uv workspace, install everything
in editable mode with:

.. code-block:: bash

   uv sync --all-packages

Source changes take effect immediately without reinstalling.

To install individual optional extras:

.. code-block:: bash

   uv sync --extra mlflow
   uv sync --extra pybamm
   uv sync --extra sdl
   uv sync --extra camd     # see camd note below for two additional pip steps


Standalone pip install
----------------------

.. code-block:: bash

   pip install "."             # core only
   pip install ".[mlflow]"     # + MLflow logging
   pip install ".[pybamm]"     # + PyBAMM demo
   pip install ".[sdl]"        # + self-driving lab demo
   # camd — see note below


Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 30

   * - Extra
     - Key packages
     - Required for
   * - ``mlflow``
     - mlflow, plotly
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


camd — legacy compatibility
----------------------------

The ``camd`` and ``qmpy-tri`` packages pin old versions of scipy and bokeh
that conflict with other extras (``pybamm`` requires ``scipy ≥ 1.11.4``).
They must be installed separately, *without* their declared dependencies:

.. code-block:: bash

   pip install ".[camd]"        # installs pymatgen, matminer, django, PuLP …
   pip install camd --no-deps   # camd itself — skips scipy-pinning GPy
   pip install qmpy-tri --no-deps  # qmpy thermodynamics — skips old bokeh pin

On first run, ``ta-camd-demo`` downloads the OQMD Voronoi-Magpie fingerprints
dataset (~150 MB) from `data.matr.io <https://data.matr.io>`_ and caches it
under ``~/.cache/traits_audit/``.  If the download fails, synthetic data
with the same schema is used automatically.


Conda environment
-----------------

Use the conda env as the base and install pip packages on top:

.. code-block:: bash

   conda activate traits-audit
   pip install -e "."          # editable install of core
   pip install -e ".[pybamm]"  # add PyBAMM extra, etc.

For the camd demo, apply the three-step install above after activating the
conda environment.


Verifying the install
---------------------

.. code-block:: python

   from traits_audit import AuditHook, AuditPipeline
   from traits_audit.checks import CalibrationErrorCheck
   print("OK")
