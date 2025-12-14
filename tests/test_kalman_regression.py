########################################################################################################################
##### IMPORTS ##########################################################################################################
########################################################################################################################

# standard library imports

# internal imports
from quadratic_kalman.unconstraint_kalman_regression import kalman_regression

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

    def test_kalman_regression(self) -> None:

        # TODO

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key=key, num=2)

        T = 128
        random_walk_std = 0.1
        noise_std = 0.1

        initial_parameter = jnp.array([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.2]
        ])
        random_walk = jnp.cumsum(random_walk_std * jax.random.normal(key=k1, shape=(T, 2, 3)), axis=0)
        true_parameters = initial_parameter[None, :, :] + random_walk

        x = jax.random.normal(key=k2, shape=(T, 2))
        y = jnp.einsum("ti, tij -> tj", x, true_parameters) + noise_std * jax.random.normal(key=k2, shape=(T, 3))

        result = kalman_regression(
            x=x, y=y,
            initial_parameter_estimate=initial_parameter,
            initial_covariance_of_parameter=0.01,
            parameter_transition_matrix=None,
            parameter_transition_noise_covariance=random_walk_std**2,
            observation_noise_covariance=noise_std**2,
            prior_importance=10.
        )

        true_final_parameter = true_parameters[-1]
        final_parameter_estimate = result["updated_parameters"][-1, ...]

        # import matplotlib.pyplot as plt
        # print(true_final_parameter)
        # print(final_parameter_estimate)
        # fig, axs = plt.subplots(nrows=2, ncols=3)
        # for i in range(2):
        #     for j in range(3):
        #         axs[i, j].plot(true_parameters[:, i, j], label=f"true {i}{j}")
        #         axs[i, j].plot(result["updated_parameters"][:, i, j], label=f"{i}{j}")
        #         axs[i, j].grid()
        #         axs[i, j].legend()
        # plt.show()

        # assert that final position estimate is close to true position
        self.assertTrue(jnp.mean(jnp.abs(true_final_parameter - final_parameter_estimate)) < 0.5)
