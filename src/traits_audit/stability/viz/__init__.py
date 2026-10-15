"""Figure generators for the stability analysis (matplotlib; PNG output).

Nothing here mutates global matplotlib state on import; every public plot
function renders under :func:`~traits_audit.stability.viz.common.style_context`.
Visual identity (label order, colors, markers) is injected with a
:class:`~traits_audit.stability.viz.style.PlotStyle`.
"""
