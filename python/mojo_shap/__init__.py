"""TreeSHAP and Kernel SHAP accelerated with Mojo."""

from ._explanation import Explanation
from ._kernel import KernelExplainer
from ._tree import TreeExplainer


def Explainer(model, masker=None, *, algorithm="auto", **kwargs):
    """Select the covered tree or kernel explainer from an upstream-like call."""
    if algorithm in ("auto", "tree") and (
        hasattr(model, "tree_") or hasattr(model, "estimators_")
    ):
        return TreeExplainer(model, data=masker, **kwargs)
    if algorithm in ("auto", "kernel") and callable(model) and masker is not None:
        return KernelExplainer(model, masker, **kwargs)
    raise TypeError("no covered explainer matches this model and masker")

__all__ = ["Explanation", "Explainer", "KernelExplainer", "TreeExplainer"]
__version__ = "0.1.0"
