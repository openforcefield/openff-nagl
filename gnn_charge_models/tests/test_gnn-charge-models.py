"""
Unit and regression test for the gnn-charge-models package.
"""

# Import package, test suite, and other packages as needed
import gnn_charge_models
import pytest
import sys


def test_gnn_charge_models_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "gnn_charge_models" in sys.modules
