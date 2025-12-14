########################################################################################################################
##### IMPORTS ##########################################################################################################
########################################################################################################################

# standard library imports

# internal imports
from quadratic_kalman.reshaping_utils import maybe_repeat_across_time
from quadratic_kalman.unconstrained_quadratic_kalman import quadratic_kalman

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
class KalmanRegressionOutput(TypedDict):
    predicted_parameters: jndarray
    predicted_parameter_covariances: jndarray
    y_hat: jndarray
    innovations: jndarray
    updated_parameters: jndarray
    updated_parameter_covariances: jndarray

########################################################################################################################
##### CONSTANTS & TODOs ################################################################################################
########################################################################################################################

# CONSTANTS

########################################################################################################################

# TODO:
#   - Covariance matrix of observations
#   - Momentum

########################################################################################################################
########################################################################################################################
########################################################################################################################


def _get_H(x: jndarray, d_x: int, d_y: int) -> jndarray:
    """
    Build observation matrix H_t from the observed x_t. We want y_t ~ x_t @ beta_t = H_t @ vec(beta_t) with beta_t of
    shape (d_x, d_y) and x_t of shape (d_x,). Here vec(-) is row-major/C-type vectorization where we have the identity
        vec(ABC) = (A ⊗ C') vec(B) and hence vec(AB) = vec(A ⊗ E) vec(B)
    for all compatible matrices A, B, C. Thus
        vec(x_t \beta_t) = (x_t ⊗ E) vec(\beta_t)
    and H_t = x_t @ E is the desired matrix. Note that x_t is considered as a (1, d_x) matrix(!) for this to work.

    :param x: of shape (d_x,),
    :param d_x: the dimensionality of x,
    :param d_y: the dimensionality of y.
    :return: the matrix H of shape (d_y, d_x * d_y) for which H @ vec(\beta) = x @ \beta for all \beta.
    """
    return jnp.kron(x.reshape(1, d_x), jnp.eye(d_y))


def _get_Hs(x: jndarray, d_y: int, d_x: int) -> jndarray:
    """
    For x of shape (T, d_x) constructs the observation matrices Hs, an array of shape (T, d_y, d_x * d_y).
    """
    return jax.vmap(lambda x: _get_H(x=x, d_x=d_x, d_y=d_y))(x)


def kalman_regression(
    x: jndarray,
    y: jndarray,
    initial_parameter_estimate: jndarray,
    initial_covariance_of_parameter: jndarray | float | None = None,
    parameter_transition_matrix: jndarray | float | None = None,
    parameter_transition_noise_covariance: jndarray | float | None = None,
    observation_noise_covariance: jndarray | float | None = None,
    prior_importance: float | None = None
) -> KalmanRegressionOutput:
    r"""
    For each point in time t, estimates a (d_x, d_y) matrix \beta_t such that y_t ~ x_t @ \beta_t.
    Here x_t and y_t are treated as column vectors, i. e. treated as if of shape (1, d_x) and (1, d_y).
    # TODO: Describe this in the TeX-document.

    :param x:
        the independent variables - of shape (T,) or (T, d_y),
    :param y:
        the dependent variables - of shape (T,) or (T, d_x),
    :param initial_parameter_estimate:
        initial guess \beta_0 - of shape (d_x, d_y)
    :param initial_covariance_of_parameter:
        covariance of initial guess of \beta_0 - broadcasted to shape (d_x, d_y, d_x, d_y),
    :param parameter_transition_matrix:
        assumed dynamics F_t such that X_t = F_t @ X_{t-1} + W_t - broadcasted to shape (T, d_x, d_y, d_x, d_y),
    :param parameter_transition_noise_covariance:
        covariance of the state transition noise W_t - broadcasted to shape (T, d_x, d_y, d_x, d_y),
    :param observation_noise_covariance:
        covariance of observation noise y_t - \hat y_t = y_t - x_t @ \beta_t - broadcasted to shape (T, d_y, d_y),
    :param prior_importance:
        acts as regularization.
    :return:
        The estimated \beta_t across time - of shape (T, d_x, d_y).
    """

    # if either x or y is one-dimensional, adjust shapes
    if len(x.shape) == 1:
        x = x[:, None]
    if len(y.shape) == 1:
        y = y[:, None]

    # assert that shapes are correct
    if len(x.shape) != 2:
        raise ValueError("Argument \'x\' must have 1 or 2 axes!")
    if len(y.shape) != 2:
        raise ValueError("Argument \'y\' must have 1 or 2 axes!")

    # get number of points in time and dimensionality of both x and y; from these calculate the state size
    T, d_x = x.shape
    _, d_y = y.shape
    state_size = d_y * d_x

    # (maybe) reshape initial estimates of parameter and its covariance
    if isinstance(initial_parameter_estimate, jndarray):
        initial_parameter_estimate = initial_parameter_estimate.reshape(state_size)
    if isinstance(initial_covariance_of_parameter, jndarray):
        initial_covariance_of_parameter = initial_covariance_of_parameter.reshape(state_size, state_size)

    # helper function to parse the "matrices" supplied by the caller
    def parse_maybe_matrix_that_is_square(
        maybe_matrix: jndarray | float | None,
        d: int,
        maybe_repeat: bool = True
    ) -> jndarray:

        if maybe_matrix is None:
            maybe_matrix = 1.

        if isinstance(maybe_matrix, float):
            maybe_matrix = maybe_matrix * jnp.eye(d)

        if maybe_repeat:
            maybe_matrix = maybe_repeat_across_time(matrix_or_matrices=maybe_matrix, repeats=T)

        return maybe_matrix

    # parse inputs to initial covariance of parameter and Fs, Qs and Rs
    initial_covariance_of_parameter = parse_maybe_matrix_that_is_square(
        maybe_matrix=initial_covariance_of_parameter,
        d=state_size,
        maybe_repeat=False
    )
    Fs = parse_maybe_matrix_that_is_square(maybe_matrix=parameter_transition_matrix, d=state_size)
    Qs = parse_maybe_matrix_that_is_square(maybe_matrix=parameter_transition_noise_covariance, d=state_size)
    Rs = parse_maybe_matrix_that_is_square(maybe_matrix=observation_noise_covariance, d=d_y)

    # build observation matrices from x
    Hs = _get_Hs(x, d_y=d_y, d_x=d_x)  # of shape (T, d_y, d_x * d_y)

    # use y and the corresponding covariances Rs to construct the quadratic objective
    # TODO: Make clear how this works
    # TODO: Can calculate H_t.T @ R_t^{-1} first and get away with a single solve
    Rs_inv_at_Hs = jax.vmap(fun=jnp.linalg.solve, in_axes=0)(Rs, Hs)  # R_t^{-1} @ H_t
    Rs_inv_at_y = jax.vmap(fun=jnp.linalg.solve, in_axes=0)(Rs, y)  # R_t^{-1} @ y_t
    hessians = jnp.einsum("...ji, ...jk -> ...ik", Hs, Rs_inv_at_Hs)  # H_t.T @ R_t^{-1} @ H_t
    gradients = -jnp.einsum("...ji, ...j -> ...i", Hs, Rs_inv_at_y)  # H_t.T @ R_t^{-1} @ y_t

    # rescaling the prior term and leaving quadratic objective fixed is equivalent to
    # rescaling the quadratic objective and leaving the prior term fixed
    if prior_importance:
        hessians *= 1/prior_importance
        gradients *= 1/prior_importance

    # apply the quadratic Kalman filter
    result = quadratic_kalman(
        initial_state_estimate=initial_parameter_estimate,
        initial_state_covariance=initial_covariance_of_parameter,
        state_transition_matrices=Fs,
        state_transition_noise_covariance=Qs,
        hessians_of_quadratic_objective=hessians,
        gradients_of_quadratic_objective=gradients
    )

    # unflatten results
    predicted_parameters = jnp.reshape(result["predicted_states"], (T, d_x, d_y))
    predicted_parameter_covariances = jnp.reshape(result["predicted_covariances"], (T, d_x, d_y, d_x, d_y))
    updated_parameter = jnp.reshape(result["predicted_states"], (T, d_x, d_y))
    updated_parameter_covariances = jnp.reshape(result["updated_covariances"], (T, d_x, d_y, d_x, d_y))

    # calculate the estimates of y_t for each t and calculate the corresponding error/innovation
    y_hat = jnp.einsum("...i, ...ij -> ...j", x, predicted_parameters)
    innovations = y - y_hat

    return {
        "predicted_parameters": predicted_parameters,
        "predicted_parameter_covariances": predicted_parameter_covariances,
        "y_hat": y_hat,
        "innovations": innovations,
        "updated_parameters": updated_parameter,
        "updated_parameter_covariances": updated_parameter_covariances,
    }
