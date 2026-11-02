Installation
============

Requirements: Python ≥ 3.11.  Core dependencies are ``numpy``, ``scipy``,
``scikit-learn`` and ``matplotlib``.


From PyPI
---------

.. code-block:: bash

   pip install traits-audit                # core only
   pip install "traits-audit[mlflow]"      # + MLflow logging
   pip install "traits-audit[stability]"   # + PyDMD windowed stability curves


Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 30

   * - Extra
     - Key packages
     - Required for
   * - ``mlflow``
     - mlflow
     - :class:`~traits_audit.mlflow_logger.MLflowLogger`
   * - ``stability``
     - pydmd
     - :mod:`traits_audit.stability.windowed` PyDMD curves
   * - ``docs``
     - sphinx, furo, myst-parser, sphinx-autodoc-typehints
     - Building this documentation
   * - ``dev``
     - pytest, pytest-cov
     - Running the test suite


From source (uv workspace)
--------------------------

The repository is a uv workspace holding the library
(``packages/traits-audit``) and the paper reproduction code
(``experiments/nmi2026``):

.. code-block:: bash

   git clone https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit.git
   cd traits-audit
   uv sync                  # library + experiment, editable, with dev tools
   uv sync --all-extras     # + the case studies' heavy dependencies

The case-study demos (``ta-cal-demo``, ``ta-sdl-demo``, ``ta-mobo-demo``,
``ta-pybamm-demo``, ``ta-camd-demo``) belong to the experiment package, not
the library. See ``experiments/nmi2026/README.md``.


Verifying the install
---------------------

.. code-block:: bash

   ta-demo
