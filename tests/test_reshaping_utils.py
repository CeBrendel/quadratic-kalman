########################################################################################################################
##### IMPORTS ##########################################################################################################
########################################################################################################################

# standard library imports

# internal imports
from quadratic_kalman.reshaping_utils import maybe_repeat_across_time

# external imports
import jax.numpy as jnp

########################################################################################################################
##### typing (general | standard | internal | external | custom types) #################################################
########################################################################################################################

# general typing

# standard library types
from unittest import TestCase

# internal types

# external types

# custom types

########################################################################################################################
##### CONSTANTS & TODOs ################################################################################################
########################################################################################################################

# CONSTANTS

########################################################################################################################

# TODO

########################################################################################################################
########################################################################################################################
########################################################################################################################


class TestMaybeRepeat(TestCase):

    def test_reshape_exceptions(self) -> None:

        n = 13
        m = 17
        o = 23
        p = 57
        matrix_1d = jnp.zeros(shape=(n,))
        matrix_4d = jnp.zeros(shape=(n, m, o, p))

        with self.assertRaises(expected_exception=ValueError):
            maybe_repeat_across_time(matrix_or_matrices=matrix_1d, repeats=4)
            maybe_repeat_across_time(matrix_or_matrices=matrix_4d, repeats=4)

    def test_reshape_shapes(self) -> None:

        n = 13
        m = 17
        o = 23
        matrix_2d = jnp.zeros(shape=(n, m))
        matrix_3d = jnp.zeros(shape=(n, m, o))

        self.assertEqual(maybe_repeat_across_time(matrix_or_matrices=matrix_2d, repeats=4).shape, (4, n, m))
        self.assertEqual(maybe_repeat_across_time(matrix_or_matrices=matrix_3d, repeats=n).shape, (n, m, o))
