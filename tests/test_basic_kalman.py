########################################################################################################################
##### IMPORTS ##########################################################################################################
########################################################################################################################

# standard library imports
import unittest

# internal imports
from quadratic_kalman.__basic_kalman_for_reference import kalman

# external imports
import jax
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


class TestKalman(TestCase):

    def test_kalman_with_fixed_velocity(self) -> None:

        key = jax.random.PRNGKey(0)

        dt = 1.0
        state_transition_matrix = jnp.array(
            [[1., dt],
             [0., 1.]]
        )
        state_transition_covariance = 0.01 * jnp.eye(2)

        observation_matrix = jnp.array([[1., 0.]])
        observation_covariance = jnp.array([[0.1**2]])

        initial_state_estimate = jnp.array([0., 1.])
        initial_covariance_estimate = jnp.eye(2)

        T = 20
        true_positions = jnp.arange(T)
        noise = 0.1 * jax.random.normal(key=key, shape=(T,))
        observations = (true_positions + noise)[..., None]

        result = kalman(
            observations=observations,
            initial_state_estimate=initial_state_estimate,
            initial_state_covariance=initial_covariance_estimate,
            observation_matrices=observation_matrix,
            state_transition_matrices=state_transition_matrix,
            observation_noise_covariance=observation_covariance,
            state_transition_noise_covariance=state_transition_covariance
        )

        true_final_position = true_positions[-1]
        final_position_estimate = result["updated_states"][-1, 0]

        # assert that final position estimate is close to true position
        self.assertTrue(abs(final_position_estimate - true_final_position) < 0.5)

    def test_kalman_with_random_acceleration(self) -> None:

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key=key, num=2)

        dt = 1.
        state_transition_matrix = jnp.array(
            [[1., dt, 0.],
             [0., 1., dt],
             [0., 0., 1.]]
        )
        observation_matrix = jnp.array([[1., 0., 0.]])

        state_transition_covariance = 0.01 * jnp.eye(3)
        observation_covariance = jnp.array([[0.1**2]])

        initial_state_estimate = jnp.array([0., 0., 0.])
        initial_covariance_estimate = jnp.eye(3)

        T = 128
        true_accelerations = 0.3 * jax.random.normal(key=k1, shape=(T,))
        true_velocities = jnp.cumsum(true_accelerations, axis=0)
        true_positions = jnp.cumsum(true_velocities, axis=0)

        observation_noise = 0.1 * jax.random.normal(key=k2, shape=(T,))
        observations = (true_positions + observation_noise)[..., None]

        result = kalman(
            observations=observations,
            initial_state_estimate=initial_state_estimate,
            initial_state_covariance=initial_covariance_estimate,
            observation_matrices=observation_matrix,
            state_transition_matrices=state_transition_matrix,
            observation_noise_covariance=observation_covariance,
            state_transition_noise_covariance=state_transition_covariance
        )

        true_final_position = true_positions[-1]
        final_position_estimate = result["updated_states"][-1, 0]

        # assert that final position estimate is close to true position
        self.assertTrue(abs(final_position_estimate - true_final_position) < 0.5)
