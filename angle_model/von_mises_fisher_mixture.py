import os

from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro

NUM_CHAINS = 4
numpyro.set_host_device_count(NUM_CHAINS)

import arviz as az
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import numpyro.infer.reparam
from numpyro.infer import Predictive

# Load generated data from rose diagram as obseration
y_obs = np.load("output/rose_diagram.npy")
y_obs = jnp.radians(y_obs)

# Required random seeds
random_seed = jnp.frombuffer(os.urandom(8), dtype=jnp.int64)[0]
# random_seed = 584479765808204282  # Seed for reproducing the results
print(f"Random seed: {random_seed}")
np.save("output/random_seed.npy", random_seed)
key, subkey = random.split(random.PRNGKey(random_seed))


@numpyro.handlers.reparam(
    config={
        "mu_1": numpyro.infer.reparam.CircularReparam(),
        "mu_2": numpyro.infer.reparam.CircularReparam(),
    }
)
def model(y_obs=None):
    kappa_1 = numpyro.sample("kappa_1", dist.Gamma(20.0, 0.1))
    kappa_2 = numpyro.sample("kappa_2", dist.Gamma(20.0, 0.1))

    # Means for generated data from rose diagram
    mu_1 = numpyro.sample(
        "mu_1", dist.VonMises(loc=jnp.radians(105), concentration=1 / jnp.radians(10) ** 2)
    )
    mu_2 = numpyro.sample(
        "mu_2", dist.VonMises(loc=jnp.radians(125), concentration=1 / jnp.radians(10) ** 2)
    )

    vm_1 = dist.VonMises(loc=mu_1, concentration=kappa_1)
    vm_2 = dist.VonMises(loc=mu_2, concentration=kappa_2)

    w = numpyro.sample("w", dist.Uniform(0.0, 1.0))
    mix = dist.Categorical(probs=jnp.array([w, 1.0 - w]))

    with numpyro.plate("y_obs", len(y_obs) if y_obs is not None else 1):
        _ = numpyro.sample("y", dist.MixtureGeneral(mix, [vm_1, vm_2]), obs=y_obs)


key, subkey = random.split(random.PRNGKey(random_seed))
del subkey

nuts_kernel = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=5000, num_samples=50000, num_chains=NUM_CHAINS)
key, *subkey = random.split(key, NUM_CHAINS + 1)
mcmc.run(jnp.array(subkey), y_obs=y_obs)
del subkey
posterior_samples = mcmc.get_samples()

posterior_predictive = Predictive(model, posterior_samples=posterior_samples)
key, subkey = random.split(key)
posterior_predictive_samples = posterior_predictive(subkey)
del subkey
np.save("output/random_rotation_angle.npy", posterior_predictive_samples["y"][::25].flatten())

print(posterior_predictive_samples["y"][::25].flatten())
print(len(posterior_predictive_samples["y"][::25].flatten()))

data = az.from_numpyro(
    posterior=mcmc,
    posterior_predictive=posterior_predictive_samples,
)
summary = az.summary(data)
print(summary)

# Save the summary to a CSV file
file_name = "output/summary.csv"
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
plt.savefig("output/fitted_model.pdf")
