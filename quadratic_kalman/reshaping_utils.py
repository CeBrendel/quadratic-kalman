########################################################################################################################
##### IMPORTS ##########################################################################################################
########################################################################################################################

# standard library imports

# internal imports

# external imports
import jax.numpy as jnp

########################################################################################################################
##### typing (general | standard | internal | external | custom types) #################################################
########################################################################################################################

# general typing

# standard library types

# internal types

# external types
from jax.numpy import ndarray as jndarray

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


def maybe_repeat_across_time(matrix_or_matrices: jndarray, repeats: int) -> jndarray:
    """
    If *matrix_or_matrices* is not already a batch of matrices, repeat it across time, otherwise return the batch.
    :param matrix_or_matrices: the matrix (or matrices) to maybe repeat,
    :param repeats: how often to maybe repeat the given matrix.
    :return: The maybe repeated matrix, or the original array.
    """

    if not isinstance(matrix_or_matrices, jndarray):
        raise ValueError("Supplied argument was not a \'jax.numpy.ndarray\'!")
    if not isinstance(repeats, int):
        raise ValueError("Supplied argument was not a \'int\'!")

    current_number_of_axes = len(matrix_or_matrices.shape)

    if not (2 <= current_number_of_axes <= 3):
        raise ValueError("Supplied array must have either 2 or 3 axes!")

    if current_number_of_axes == 3:
        return matrix_or_matrices
    else:
        return jnp.repeat(matrix_or_matrices[None, ...], repeats=repeats, axis=0)
