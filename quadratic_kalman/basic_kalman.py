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
class FilterOutput(TypedDict):
    predicted_states: jndarray
    predicted_covariances: jndarray
    predicted_observations: jndarray
    innovations: jndarray
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


def kalman(
    observations: jndarray,
    initial_state_estimate: jndarray,
    initial_state_covariance: jndarray,
    observation_matrices: jndarray,
    state_transition_matrices: jndarray,
    observation_noise_covariance: jndarray,
    state_transition_noise_covariance: jndarray
) -> FilterOutput:
    """
    Estimates the expectation and covariance of the state at each point in time
    (both a priori and a posteriori) using the supplied observations.

    :param observations: array of shape (T, d_obs),
    :param initial_state_estimate: array of shape (d_state,),
    :param initial_state_covariance: array of shape (d_state, d_state),
    :param observation_matrices: array of shape (d_obs, d_state),
    :param state_transition_matrices: array of shape (d_state, d_state),
    :param observation_noise_covariance: array of shape (d_obs, d_obs),
    :param state_transition_noise_covariance: array of shape (d_state, d_state).

    :return: The estimates of a priori and a posteriori expectation and covariance of state, as well as the innovations.
    """

    # Note: This implementation closely follows https://en.wikipedia.org/wiki/Kalman_filter and follows its notation.
    # The article was accessed at 29.11.2025, 18:58

    # get relevant counts
    T, d_obs = observations.shape
    d_state, = initial_state_estimate.shape

    # get Hs, Fs, Qs and Rs and repeat them across time if needed
    Hs = maybe_repeat_across_time(observation_matrices, repeats=T)
    Fs = maybe_repeat_across_time(state_transition_matrices, repeats=T)
    Qs = maybe_repeat_across_time(state_transition_noise_covariance, repeats=T)
    Rs = maybe_repeat_across_time(observation_noise_covariance, repeats=T)

    # a function that will be iterated by jax.lax.scan for each point in time
    def step(carry, inputs):

        # unpack carry and inputs
        x_prev, P_prev = carry
        y_t, H_t, F_t, Q_t, R_t = inputs

        # prediction step
        x_predicted = F_t @ x_prev
        P_predicted = F_t @ P_prev @ F_t.T + Q_t
        y_predicted = H_t @ x_predicted

        # enforce symmetry (for numerical stability)
        P_predicted: jndarray = 0.5 * (P_predicted + P_predicted.T)

        # get innovation and innovation covariance
        innovation = y_t - y_predicted
        S = H_t @ P_predicted @ H_t.T + R_t

        # calculate optimal Kalman gain K = P_predicted @ H_t.T @ S^{-1}, notice that solving
        # K @ S = P_predicted @ H_t.T is equivalent to solving S.T @ K.T = H_t @ P_predicted.T
        K = jnp.linalg.solve(S.T, H_t @ P_predicted.T).T

        # update step, TODO: Joseph form of update guarantees positive semi-definiteness of P_updated
        x_updated = x_predicted + K @ innovation
        P_updated = (jnp.eye(d_state) - K @ H_t) @ P_predicted

        # enforce symmetry (for numerical stability)
        P_updated: jndarray = 0.5 * (P_updated + P_updated.T)

        # return results
        carry = (x_updated, P_updated)
        outputs = (x_predicted, P_predicted, x_updated, P_updated, y_predicted, innovation)
        return carry, outputs

    # get initial input and carry for use in jax.lax.scan
    inputs = (observations, Hs, Fs, Qs, Rs)
    init_carry = (initial_state_estimate, initial_state_covariance)

    # perform scan
    _, outputs = jax.lax.scan(step, init_carry, inputs, length=T)
    xs_predicted, Ps_predicted, xs_updated, Ps_updated, ys_predicted, innovations = outputs

    return {
        "predicted_states": xs_predicted,
        "predicted_covariances": Ps_predicted,
        "predicted_observations": innovations,
        "innovations": innovations,
        "updated_states": xs_updated,
        "updated_covariances": Ps_updated
    }
