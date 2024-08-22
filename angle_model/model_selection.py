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
# random_seed = -4980610957694664259  # Seed for reproducing the results
print(random_seed)
np.save("output/random_seed_model_selection.npy", random_seed)


# --- Model 1 ---

@numpyro.handlers.reparam(config={"mu": numpyro.infer.reparam.CircularReparam()})
def model1(y_obs=None):
    kappa = numpyro.sample("kappa", dist.Gamma(20.0, 0.1))

    # Non-informative prior
    mu = numpyro.sample(
        "mu", dist.VonMises(loc=jnp.radians(0), concentration=1 / jnp.radians(180) ** 2)
    )

    with numpyro.plate("y_obs", len(y_obs) if y_obs is not None else 1):
        _ = numpyro.sample("y", dist.VonMises(loc=mu, concentration=kappa), obs=y_obs)


nuts_kernel1 = numpyro.infer.NUTS(model1)
mcmc1 = numpyro.infer.MCMC(
    nuts_kernel1, num_warmup=num_warmup, num_samples=num_samples, num_chains=NUM_CHAINS
)
key = random.PRNGKey(random_seed)
key, *subkey = random.split(key, NUM_CHAINS + 1)
mcmc1.run(jnp.array(subkey), y_obs=y_obs)
del subkey
posterior_samples1 = mcmc1.get_samples()

posterior_predictive1 = Predictive(model1, posterior_samples=posterior_samples1)
key, subkey = random.split(key)
posterior_predictive_samples1 = posterior_predictive1(subkey)
del subkey

data1 = az.from_numpyro(
    posterior=mcmc1,
    posterior_predictive=posterior_predictive_samples1,
)
summary1 = az.summary(data1)
print(summary1)


# --- Model 2 ---

@numpyro.handlers.reparam(
    config={
        "mu_1": numpyro.infer.reparam.CircularReparam(),
        "mu_2": numpyro.infer.reparam.CircularReparam(),
    }
)
def model2(y_obs=None):
    kappa_1 = numpyro.sample("kappa_1", dist.Gamma(20.0, 0.1))
    kappa_2 = numpyro.sample("kappa_2", dist.Gamma(20.0, 0.1))

    # Non-informative prior
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


nuts_kernel2 = numpyro.infer.NUTS(model2)
mcmc2 = numpyro.infer.MCMC(
    nuts_kernel2, num_warmup=num_warmup, num_samples=num_samples, num_chains=NUM_CHAINS
)
key, *subkey = random.split(key, NUM_CHAINS + 1)
mcmc2.run(jnp.array(subkey), y_obs=y_obs)
del subkey
posterior_samples2 = mcmc2.get_samples()

posterior_predictive2 = Predictive(model2, posterior_samples=posterior_samples2)
key, subkey = random.split(key)
posterior_predictive_samples2 = posterior_predictive2(subkey)
del subkey

data2 = az.from_numpyro(
    posterior=mcmc2,
    posterior_predictive=posterior_predictive_samples2,
)
summary2 = az.summary(data2)
print(summary2)


# --- Model 3 ---

@numpyro.handlers.reparam(
    config={
        "mu_01": numpyro.infer.reparam.CircularReparam(),
        "mu_02": numpyro.infer.reparam.CircularReparam(),
        "mu_03": numpyro.infer.reparam.CircularReparam(),
    }
)
def model3(y_obs=None):
    kappa_01 = numpyro.sample("kappa_01", dist.Gamma(20.0, 0.1))
    kappa_02 = numpyro.sample("kappa_02", dist.Gamma(20.0, 0.1))
    kappa_03 = numpyro.sample("kappa_03", dist.Gamma(20.0, 0.1))

    # Non-informative prior
    mu_01 = numpyro.sample(
        "mu_01", dist.VonMises(loc=jnp.radians(0), concentration=1 / jnp.radians(180) ** 2)
    )
    mu_02 = numpyro.sample(
        "mu_02", dist.VonMises(loc=jnp.radians(0), concentration=1 / jnp.radians(180) ** 2)
    )
    mu_03 = numpyro.sample(
        "mu_03", dist.VonMises(loc=jnp.radians(0), concentration=1 / jnp.radians(180) ** 2)
    )

    vm_01 = dist.VonMises(loc=mu_01, concentration=kappa_01)
    vm_02 = dist.VonMises(loc=mu_02, concentration=kappa_02)
    vm_03 = dist.VonMises(loc=mu_03, concentration=kappa_03)

    mix_weights = numpyro.sample("mix_weights", dist.Dirichlet(jnp.ones((3,))))
    mix_0 = dist.Categorical(mix_weights)

    with numpyro.plate("y_obs", len(y_obs) if y_obs is not None else 1):
        _ = numpyro.sample("y", dist.MixtureGeneral(mix_0, [vm_01, vm_02, vm_03]), obs=y_obs)


nuts_kernel3 = numpyro.infer.NUTS(model3)
mcmc3 = numpyro.infer.MCMC(
    nuts_kernel3, num_warmup=num_warmup, num_samples=num_samples, num_chains=NUM_CHAINS
)
key, *subkey = random.split(key, NUM_CHAINS + 1)
mcmc3.run(jnp.array(subkey), y_obs=y_obs)
del subkey
posterior_samples3 = mcmc3.get_samples()

posterior_predictive3 = Predictive(model3, posterior_samples=posterior_samples3)
key, subkey = random.split(key)
posterior_predictive_samples3 = posterior_predictive3(subkey)
del subkey

data3 = az.from_numpyro(
    posterior=mcmc3,
    posterior_predictive=posterior_predictive_samples3,
)
summary3 = az.summary(data3)
print(summary3)


# --- Comparison ---

waic1 = az.waic(data1, var_name="y")
print(waic1)
waic2 = az.waic(data2, var_name="y")
print(waic2)
waic3 = az.waic(data3, var_name="y")
print(waic3)

print("Compare results:")
df_comp_loo = az.compare(
    {
        "simple_model": data1,
        "mixture_model_2vm": data2,
        "mixture_model_3vm": data3,
    },
    var_name="y",
)
print(df_comp_loo)
df_comp_waic = az.compare(
    {
        "simple_model": data1,
        "mixture_model_2vm": data2,
        "mixture_model_3vm": data3,
    },
    var_name="y",
    ic="waic",
)
print(df_comp_waic)

# Plot posterior predictive samples for model 1
fig1, ax1 = plt.subplots(figsize=(4, 3))
ax1.hist(
    posterior_predictive_samples1["y"][::20].flatten(),
    density=True,
    bins=30,
    alpha=0.5,
    label="Posterior predictive (Simple)",
)
ax1.hist(y_obs.flatten(), density=True, bins=30, alpha=0.5, label="Observed data")
ax1.set_xlabel(r'Rotation angle (rad)')
ax1.set_ylabel("Density")
ax1.set_xlim(1.3, 2.7)
ax1.set_ylim(0.0, 4.2)
ax1.legend()
fig1.savefig("output/posterior_predictive_compare_models_simple.pdf")

# Plot posterior predictive samples for model 2
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.hist(
    posterior_predictive_samples2["y"][::20].flatten(),
    density=True,
    bins=30,
    alpha=0.5,
    label="Posterior predictive (2VM)",
)
ax2.hist(y_obs.flatten(), density=True, bins=30, alpha=0.5, label="Observed data")
ax2.set_xlabel(r'Rotation angle (rad)')
ax2.set_ylabel("Density")
ax2.set_xlim(1.3, 2.7)
ax2.set_ylim(0.0, 4.2)
ax2.legend()
fig2.savefig("output/posterior_predictive_compare_models_2vm.pdf")

# Plot posterior predictive samples for model 3
fig3, ax3 = plt.subplots(figsize=(4, 3))
ax3.hist(
    posterior_predictive_samples3["y"][::20].flatten(),
    density=True,
    bins=30,
    alpha=0.5,
    label="Posterior predictive (3VM)",
)
ax3.hist(y_obs.flatten(), density=True, bins=30, alpha=0.5, label="Observed data")
ax3.set_xlabel(r'Rotation angle (rad)')
ax3.set_ylabel("Density")
ax3.set_xlim(1.3, 2.7)
ax3.set_ylim(0.0, 4.2)
ax3.legend()
fig3.savefig("output/posterior_predictive_compare_models_3vm.pdf")

az.plot_compare(df_comp_loo, insample_dev=False)
plt.gcf().set_size_inches(10, 6)  # Adjust size as needed
plt.savefig("output/compare_models.pdf", bbox_inches="tight")

az.plot_elpd(
    {"simple_model": data1, "mixture_model_2vm": data2, "mixture_model_3vm": data3},
    var_name="y",
    xlabels=True,
)
plt.savefig("output/elpd_compare_model.pdf")
