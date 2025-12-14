########################################################################################################################
##### IMPORTS ##########################################################################################################
########################################################################################################################

# standard library imports

# internal imports
from quadratic_kalman.reshaping_utils import maybe_repeat_across_time

# external imports
import jax
import jax.numpy as jnp

########################################################################################################################
##### typing (general | standard | internal | external | custom types) #################################################
########################################################################################################################

# general typing
from typing import TypedDict

# standard library types

# internal types

# external types
from jax.numpy import ndarray as jndarray


# custom types
class QuadraticKalmanOutput(TypedDict):
    predicted_states: jndarray
    predicted_covariances: jndarray
    updated_states: jndarray
    updated_covariances: jndarray

########################################################################################################################
##### CONSTANTS & TODOs ################################################################################################
########################################################################################################################

# CONSTANTS

########################################################################################################################

# TODO

########################################################################################################################
########################################################################################################################
########################################################################################################################


def quadratic_kalman(
    initial_state_estimate: jndarray,
    initial_state_covariance: jndarray,
    state_transition_matrices: jndarray,
    state_transition_noise_covariance: jndarray,
    gradients_of_quadratic_objective: jndarray,
    hessians_of_quadratic_objective: jndarray
) -> QuadraticKalmanOutput:
    """
    Estimates the state of the specified system using "Quadratic Kalman", see ../tex_files/main.pdf for details.

    :param initial_state_estimate: of shape (d,),
    :param initial_state_covariance: of shape (d, d),
    :param state_transition_matrices: of shape (T, d, d),
    :param state_transition_noise_covariance: of shape (T, d, d),
    :param gradients_of_quadratic_objective: of shape (T, d),
    :param hessians_of_quadratic_objective: of shape (T, d, d).
    :return:
        The estimates x_{t|t-1}, P_{t|t-1} and x_{t|t}, P_{t|t} of states and state covariances batched across time.
    """

    # assert that hessians and gradients are batched (as they replace observations)
    if len(hessians_of_quadratic_objective.shape) != 3:
        raise ValueError("Argument \'hessians_of_quadratic_objective\' must have 3 axes!")
    if len(gradients_of_quadratic_objective.shape) != 2:
        raise ValueError("Argument \'gradients_of_quadratic_objective\' must have 2 axes!")

    # get number of points in time and dimensionality of state
    T, d = gradients_of_quadratic_objective.shape

    # get Fs, Qs, hessians and gradients and repeat them across time if needed
    Fs = maybe_repeat_across_time(state_transition_matrices, repeats=T)
    Qs = maybe_repeat_across_time(state_transition_noise_covariance, repeats=T)
    hessians = hessians_of_quadratic_objective
    gradients = gradients_of_quadratic_objective

    # a function that will be iterated by jax.lax.scan for each point in time
    def step(carry, inputs):

        # unpack carry and inputs
        x_prev, P_prev = carry
        F_t, Q_t, hessian_t, gradient_t = inputs

        # prediction step
        x_predicted = F_t @ x_prev
        P_predicted = F_t @ P_prev @ F_t.T + Q_t

        # enforce symmetry (for numerical stability)
        P_predicted: jndarray = 0.5 * (P_predicted + P_predicted.T)

        # get gradient and Hessian of the composite objective (predicted prior + user-specified quadratic objective)
        precision_predicted = jnp.linalg.solve(P_predicted, jnp.eye(d))
        total_gradient = gradient_t - precision_predicted @ x_predicted
        total_hessian = hessian_t + precision_predicted

        # update step
        x_updated = -jnp.linalg.solve(total_hessian, total_gradient)
        P_updated = jnp.linalg.solve(total_hessian, jnp.eye(d))

        # enforce symmetry (for numerical stability)
        P_updated = 0.5 * (P_updated + P_updated.T)

        # return results
        carry = (x_updated, P_updated)
        outputs = (x_predicted, P_predicted, x_updated, P_updated)

        return carry, outputs

    # get initial input and carry for use in jax.lax.scan
    inputs = (Fs, Qs, hessians, gradients)
    init_carry = (initial_state_estimate, initial_state_covariance)

    # perform scan
    _, outputs = jax.lax.scan(step, init_carry, inputs, length=T)
    xs_predicted, Ps_predicted, xs_updated, Ps_updated = outputs

    return {
        "predicted_states": xs_predicted,
        "predicted_covariances": Ps_predicted,
        "updated_states": xs_updated,
        "updated_covariances": Ps_updated
    }
