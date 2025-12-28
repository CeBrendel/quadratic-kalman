########################################################################################################################
##### IMPORTS ##########################################################################################################
########################################################################################################################

# standard library imports

# internal imports
from quadratic_kalman.unconstraint_kalman_regression import kalman_regression

# external imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

########################################################################################################################
##### typing (general | standard | internal | external | custom types) #################################################
########################################################################################################################

# general typing

# standard library types

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

# fixed randomness
key = jax.random.PRNGKey(seed=0)
k1, k2, k3, k4 = jax.random.split(key=key, num=4)

# define the initial beta for which Y_0 ~ X_0 @ beta
initial_beta = jnp.asarray([
    [0., 1., -1.1],
    [1., -.5, 0.]
])

# generate a 2x3 random walk comprised of independent Brownian motions
T = 512
random_walk = jnp.cumsum(jax.random.normal(k1, shape=(T, 2, 3)), axis=0)

# add (rescaled) random walk to initial beta to obtain a time-varying beta, so that Y_t ~ X_t @ beta_t
step_std = 0.5
betas = initial_beta[None, :, :] + step_std * random_walk

# generate some data such that Y_t ~ X_t @ beta_t for each t
noise_std = 0.1
x = jax.random.normal(k2, shape=(T, 2))
y = jnp.einsum("ti, tij -> tj", x, betas) + noise_std * jax.random.normal(k4, shape=(T, 3))

# run filter
result = kalman_regression(
    y=y, x=x,
    initial_parameter_estimate=jnp.zeros_like(initial_beta),  # guess
    initial_covariance_of_parameter=1.0,  # guess
    parameter_transition_matrix=None,
    parameter_transition_noise_covariance=step_std**2,
    observation_noise_covariance=noise_std**2,
    prior_importance=1
)
estimated_betas = result["predicted_parameters"]
covariances_ast = result["predicted_parameter_covariances"]

# plot final result
fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True)
fig.suptitle("True and estimated β_t with 3σ_t confidence band")

time = jnp.arange(T)
for row_idx in range(2):
    for col_idx in range(3):
        ax = axs[row_idx, col_idx]

        # plot true beta
        ax.plot(time, betas[:, row_idx, col_idx], label=f"true β_{row_idx, col_idx}", alpha=0.9)

        # plot estimated beta with confidence interval
        relevant_beta = estimated_betas[:, row_idx, col_idx]
        relevant_std = jnp.sqrt(covariances_ast[:, row_idx, col_idx, row_idx, col_idx])
        up = relevant_beta + 3 * relevant_std
        down = relevant_beta - 3 * relevant_std
        ax.plot(time, relevant_beta, label=f"estimated β_{row_idx, col_idx}", alpha=0.8)
        ax.fill_between(
            x=time, y1=up, y2=down,
            color="green",
            alpha=0.2,
            label="3σ_t interval around estimate",
            zorder=-1
        )

        ax.grid()
        ax.legend()

plt.show()
