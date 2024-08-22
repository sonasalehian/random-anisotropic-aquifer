import os

import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer.reparam
import scienceplots
from rose_diagram import plot_rose_diagram, extract_bar_parameters
from jax import config
from numpyro.infer import Predictive

plt.style.use(['science'])

# Disable LaTeX rendering to avoid missing font issues
plt.rcParams['text.usetex'] = False

config.update("jax_enable_x64", True)
NUM_CHAINS = 4
numpyro.set_host_device_count(NUM_CHAINS)

# parameters:
num_warmup = 2000
num_samples = 20000
num_models = 4

# # Load generated data from rose diagram as observation
y_obs = np.load("output/rose_diagram.npy")
y_obs = jnp.radians(y_obs)

# Required random seeds
random_seed = jnp.frombuffer(os.urandom(8), dtype=jnp.int64)[0]
# random_seed = -167652586371646984  # Seed for reproducing the results
print(random_seed)
np.save("output/random_seed_model_selection.npy", random_seed)

@numpyro.handlers.reparam(
    config={
        "mu_1": numpyro.infer.reparam.CircularReparam(),
        "mu_2": numpyro.infer.reparam.CircularReparam(),
    }
)
def model(y_obs=None):
    kappa_1 = numpyro.sample("kappa_1", dist.Gamma(20.0, 0.1))
    kappa_2 = numpyro.sample("kappa_2", dist.Gamma(20.0, 0.1))

    mu_1 = numpyro.sample(
        "mu_1", dist.VonMises(loc=jnp.radians(0), concentration=1 / jnp.radians(180) ** 2)
    )
    mu_2 = numpyro.sample(
        "mu_2", dist.VonMises(loc=jnp.radians(0), concentration=1 / jnp.radians(180) ** 2)
    )

    vm_1 = dist.VonMises(loc=mu_1, concentration=kappa_1)
    vm_2 = dist.VonMises(loc=mu_2, concentration=kappa_2)

    mix_weights = numpyro.sample("mix_weights", dist.Dirichlet(jnp.ones((2,))))
    mix = dist.Categorical(mix_weights)

    with numpyro.plate("y_obs", len(y_obs) if y_obs is not None else 1):
        _ = numpyro.sample("y", dist.MixtureGeneral(mix, [vm_1, vm_2]), obs=y_obs)

graph = numpyro.render_model(
    model=model,
    model_args=(y_obs,),
    render_distributions=True,
    render_params=True,
     filename="output/dag-model.pdf"
)
graph

nuts_kernel = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(
    nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=NUM_CHAINS
)
key = random.PRNGKey(random_seed)
key, *subkey = random.split(key, NUM_CHAINS + 1)
mcmc.run(jnp.array(subkey), y_obs=y_obs)
del subkey
posterior_samples = mcmc.get_samples()

posterior_predictive = Predictive(model, posterior_samples=posterior_samples)
key, subkey = random.split(key)
posterior_predictive_samples = posterior_predictive(subkey)
del subkey

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
plt.savefig("output/plot_trace_2vm.pdf")

# --- save smaples of mixture of 2 vm model ---
random_rotation_angles = posterior_predictive_samples["y"][::10].flatten()
print(len(random_rotation_angles))
np.save("output/random_rotation_angle.npy", random_rotation_angles)

# Plot rose diagram of random rotation angles
theta, count = extract_bar_parameters(random_angles=random_rotation_angles)
plot_rose_diagram(theta, count)
plt.savefig('output/rose_diagram_random_rotation_angle.pdf')
