#test_fields

#
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


import numpy as np
from tokamaksim.fields import B_field, R0, a, B0, q 

def test_magnetic_axis_field():
    """Verify that the magnetic field on the axis (r=0) matches B0 dans has no poloidal component."""
    position_axis = np.array([R0, 0.0, 0.0])
    B_axis = B_field(position_axis)

    # The norm should be exactly B0 on the axis
    assert abs(np.linalg.norm(B_axis) - B0) < 10** - 10 , f"Expected B0 = {B0}, found norm = {np.linalg.norm(B_axis)}"

def test_toroidal_field_decay():
    """"Verify the 1/ R decay law of the toroidal magnetic field at internal, center, and external position"""
    for R in [R0 - a, R0, R0 + a ]:
        position = np.array([R, 0.0 , 0.0])
        B = B_field(position)
        expected_B_toroidal = B0 * R0 / R 

        #B[1] corresponds to the toroidal component ( Y axis in your cartesian projection at y = 0 )
        assert abs (B[1] - expected_B_toroidal )< 10 ** - 10,  f"Toroidal field mismatch at R={R}: expected {expected_B_toroidal}, found {B[1]}"


def test_poloidal_field_profile():
    """Verify the poloidal field magnitude profile across different minor radii (r)."""
    for r in [0.0, 0.5 * a, a]:
        position = np.array([R0 + r, 0.0, 0.0])
        B = B_field(position)
        R_local = R0 + r
        expected_B_poloidal = (r / q) * (B0 / R_local)
        
        # B[2] corresponds to the poloidal/vertical component (Z axis in your Cartesian projection at y=0)
        assert abs(B[2] - expected_B_poloidal) < 1e-10, f"Poloidal field mismatch at r={r}: expected {expected_B_poloidal}, found {B[2]}"