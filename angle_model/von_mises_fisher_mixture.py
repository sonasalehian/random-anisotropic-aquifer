import os
from jax import config

config.update("jax_enable_x64", True)
import jax.random as random

import jax.numpy as jnp
import numpy as np

import numpyro

NUM_CHAINS = 4
numpyro.set_host_device_count(NUM_CHAINS)

import numpyro.distributions as dist
import numpyro.infer.reparam
from numpyro.infer import Predictive

import arviz as az

import matplotlib.pyplot as plt


# Load generated data from rose diagram as obseration
y_obs = np.load('../output/data/generated_data_from_rose_diagram.npy')
y_obs = jnp.radians(y_obs)

# Required random seeds
random_seed = jnp.frombuffer(os.urandom(8), dtype=jnp.int64)[0]
# random_seed = 584479765808204282  # Seed for reproducing the results
print(random_seed)
np.save('../output/data/random_seed_mixture_model.npy', random_seed)
key, subkey = random.split(random.PRNGKey(random_seed))

# # Generate data from prior model
# def data_generating_model(y_obs=None):
#     kappa_1 = numpyro.sample("kappa_1", dist.Gamma(20.0, 0.1))
#     kappa_2 = numpyro.sample("kappa_2", dist.Gamma(20.0, 0.1))

#     mu_1 = numpyro.sample("mu_1", dist.VonMises(loc=jnp.radians(40), concentration=1/jnp.radians(5)**2))
#     mu_2 = numpyro.sample("mu_2", dist.VonMises(loc=jnp.radians(110), concentration=1/jnp.radians(5)**2))
    
#     vm_1 = dist.VonMises(loc=mu_1, concentration=kappa_1)
#     vm_2 = dist.VonMises(loc=mu_2, concentration=kappa_2)

#     w = numpyro.sample("w", dist.Uniform(0.0, 1.0))
#     mix = dist.Categorical(probs=jnp.array([w, 1.0 - w]))

#     with numpyro.plate("y_obs", len(y_obs) if y_obs is not None else 1):
#         y = numpyro.sample("y", dist.MixtureGeneral(mix, [vm_1, vm_2]), obs=y_obs)

# prior_predictive = Predictive(data_generating_model, num_samples=50)
# y_obs = prior_predictive(subkey)["y"]
# np.save('../output/data/generated_data_mixture_model.npy', y_obs)
del subkey

@numpyro.handlers.reparam(
    config={"mu_1": numpyro.infer.reparam.CircularReparam(), 
            "mu_2": numpyro.infer.reparam.CircularReparam(), 
            })
def model(y_obs=None):
    kappa_1 = numpyro.sample("kappa_1", dist.Gamma(20.0, 0.1))
    kappa_2 = numpyro.sample("kappa_2", dist.Gamma(20.0, 0.1))

    # Means for generated data with prior
    # mu_1 = numpyro.sample("mu_1", dist.VonMises(loc=jnp.radians(45), concentration=1/jnp.radians(5)**2))
    # mu_2 = numpyro.sample("mu_2", dist.VonMises(loc=jnp.radians(105), concentration=1/jnp.radians(5)**2))

    # Means for generated data from rose diagram
    mu_1 = numpyro.sample("mu_1", dist.VonMises(loc=jnp.radians(105), concentration=1/jnp.radians(10)**2))
    mu_2 = numpyro.sample("mu_2", dist.VonMises(loc=jnp.radians(125), concentration=1/jnp.radians(10)**2))
    
    vm_1 = dist.VonMises(loc=mu_1, concentration=kappa_1)
    vm_2 = dist.VonMises(loc=mu_2, concentration=kappa_2)

    w = numpyro.sample("w", dist.Uniform(0.0, 1.0))
    mix = dist.Categorical(probs=jnp.array([w, 1.0 - w]))

    with numpyro.plate("y_obs", len(y_obs) if y_obs is not None else 1):
        _ = numpyro.sample("y", dist.MixtureGeneral(mix, [vm_1, vm_2]), obs=y_obs)

key, subkey = random.split(random.PRNGKey(random_seed))
del subkey

nuts_kernel = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(
    nuts_kernel, num_warmup=5000, num_samples=50000, num_chains=NUM_CHAINS
)
key, *subkey = random.split(key, NUM_CHAINS + 1)
mcmc.run(jnp.array(subkey), y_obs=y_obs)
del subkey
posterior_samples = mcmc.get_samples()

posterior_predictive = Predictive(model, posterior_samples=posterior_samples)
key, subkey = random.split(key)
posterior_predictive_samples = posterior_predictive(subkey)
np.save('../output/data/random_rotation_angle.npy', posterior_predictive_samples["y"][::25].flatten())

print(posterior_predictive_samples["y"][::25].flatten())
print(len(posterior_predictive_samples["y"][::25].flatten()))
del subkey

data = az.from_numpyro(
    posterior=mcmc,
    posterior_predictive=posterior_predictive_samples,
)
summary = az.summary(data)
print(summary)

# Save the summary to a CSV file
file_name = "../output/data/vonmises_mixture_summary_mcmc.csv"
np.savetxt(file_name, summary, delimiter=",")

# Plot mu_1 and mu_2 distributions
az.plot_trace(data, var_names=["mu_1", "mu_2"])
plt.show()

plt.hist(
    posterior_predictive_samples["y"][::50].flatten(),
    density=True,
    bins=30,
    alpha=0.5,
    label="posterior predictive",
)
plt.hist(y_obs.flatten(), density=True, bins=30, alpha=0.5, label="observed data")
plt.title("Mixture of two von-Mises model")
plt.xlabel("Rotation angle")
plt.ylabel("Density")
plt.legend()
plt.xlim(1.3, 2.7)
plt.ylim(0.0, 4.2)
plt.tight_layout()
plt.show()
