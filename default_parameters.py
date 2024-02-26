# JSH: Change this to be a function create_default_parameters that returns the
# dictionary.
# Parameters

parameters = {}

# Output file name
parameters["output_dir"] = 'output/AJ/'

# Define dimention
parameters["gdim"] = 3

# Define spacial parameters
parameters["Lx"] = 2000  # Length of aquifer in x direction
parameters["Ly"] = 3500  # Length of aquifer in y direction
parameters["Lxw"] = 750  # Distance of well in x direction
parameters["Lyw"] = 1450  # Distance of well in y direction
parameters["Lz1"] = 40  # Depth of first layer of aquifer system
parameters["Lz2"] = 180  # Depth of second layer of aquifer system
parameters["Lz3"] = 180  # Depth of third layer of aquifer system
parameters["Ld1"] = 150  # Depth of the dry part of pumping well
parameters["Ld2"] = 30  # Depth of the pumping part of pumping well
parameters["Lr"] = 7  # 3.38  # radius of pumping well

# T = s
parameters["beta_s"] = 1.1E-11  # solid grain compressibility (pa^(-1))
parameters["phi"] = 0.32  # medium porosity(-)
parameters["beta_f"] = 4.4E-10  # fluid compresibility (pa^(-1))
parameters["alpha"] = 0.998  # Biot coeficient (-)
parameters["mu_f"] = 0.001  # fluid viscosity (Pa.s)
parameters["rhog"] = 9807.0  # fluid specific weight (N/m^3)
parameters["P_r"] = 0.07  # pumping rate (m^3/s)

# Define temporal parameters (s)
parameters["t"] = 0  # Start time
parameters["T"] = 345600  # (4d) #2073600 # (24d) # Final pumping time
parameters["num_steps"] = 360
parameters["T2"] = 1728000  # (20d) # Final time
parameters["num_steps2"] = 360

# Aquitard layer parameters
parameters["alpha_aqtrd"] = 0.868  # Biot coeficient (-)
parameters["lmbda_aqtrd"] = 6.088E9  # Lame's parameter (pa)
parameters["G_aqtrd"] = 7.7E9  # shear modulus (pa)
parameters["k_x_aqtrd"] = 5E-15  # medium permeability field (m^2)
parameters["k_y_aqtrd"] = 5E-15  # medium permeability field (m^2)
parameters["k_z_aqtrd"] = 5E-15  # medium permeability field (m^2)
parameters["S_e_aqtrd"] = 0.8E-10  # specific storage (Pa^(-1))

# Aquifer layer parameters
parameters["alpha_aqfr"] = 0.998  # Biot coeficient (-)
parameters["lmbda_aqfr"] = 3.768E9  # Lame's parameter (pa)
parameters["G_aqfr"] = 5.06E9  # shear modulus (pa)
parameters["k_aqfr"] = 5E-13  # medium permeability field (m^2)
parameters["k_x_aqfr"] = 1.1E-11  # medium permeability field (m^2)
parameters["k_xy_aqfr"] = 0  # medium permeability field (m^2)
parameters["k_yx_aqfr"] = 0  # medium permeability field (m^2)
parameters["k_y_aqfr"] = 4.7E-13  # medium permeability field (m^2)
parameters["k_z_aqfr"] = 5E-10  # medium permeability field (m^2)
parameters["S_e_aqfr"] = 1.5E-10  # specific storage (Pa^(-1))

# Bedrock layer parameters
parameters["alpha_bed"] = 0.858  # Biot coeficient (-)
parameters["lmbda_bed"] = 6.156E9  # Lame's parameter (pa)
parameters["G_bed"] = 7.9E9  # shear modulus (pa)
parameters["k_x_bed"] = 5E-15  # medium permeability field (m^2)
parameters["k_y_bed"] = 5E-15  # medium permeability field (m^2)
parameters["k_z_bed"] = 5E-15  # medium permeability field (m^2)
parameters["S_e_bed"] = 0.8E-10  # specific storage (Pa^(-1))

# # T = min
# parameters["beta_s"] = 2.8E-15  # solid grain compressibility (m.min^2/kg)
# parameters["phi"] = 0.32  # medium porosity(%)
# parameters["beta_f"] = 1.2E-13  # fluid compresibility (m.min^2/kg)
# parameters["alpha"] = 0.998  # Biot coeficient (-)
# parameters["mu_f"] = 0.06  # fluid viscosity (kg/(m.min))
# parameters["rhog"] = 35.3E6  # fluid specific weight (kg/(m^2.min^2))
# parameters["P_r"] = 4.2  # pumping rate (m^3/min)

# # Define temporal parameters (min)
# parameters["t"] = 0  # Start time
# parameters["T"] = 5760  # (4d) # 34560  # (24d) # Final time
# parameters["num_steps"] = 360

# # Aquitard layer parameters
# parameters["lmbda_aqtrd"] = 1.36E13  # Lame's parameter (kg/(m.min^2))
# parameters["G_aqtrd"] = 1.8E13  # shear modulus (kg/(m.min^2))
# parameters["k_x_aqtrd"] = 1.1E-11  # medium permeability field (m^2)
# parameters["k_y_aqtrd"] = 4.7E-13  # medium permeability field (m^2)
# parameters["k_z_aqtrd"] = 5E-10  # medium permeability field (m^2)
# parameters["S_e_aqtrd"] = 4.1E-14  # specific storage (m.min^2/kg)

# # Aquifer layer parameters
# parameters["lmbda_aqfr"] = 1.36E13  # Lame's parameter (kg/(m.min^2))
# parameters["G_aqfr"] = 1.8E13  # shear modulus (kg/(m.min^2))
# parameters["k_x_aqfr"] = 1.1E-11  # medium permeability field (m^2)
# parameters["k_y_aqfr"] = 4.7E-13  # medium permeability field (m^2)
# parameters["k_z_aqfr"] = 5E-10  # medium permeability field (m^2)
# parameters["S_e_aqfr"] = 2.2E-14  # specific storage (m.min^2/kg)

# # Bedrock layer parameters
# parameters["lmbda_bed"] = 1.36E13  # Lame's parameter (kg/(m.min^2))
# parameters["G_bed"] = 1.8E13  # shear modulus (kg/(m.min^2))
# parameters["k_x_bed"] = 1.1E-11  # medium permeability field (m^2)
# parameters["k_y_bed"] = 4.7E-13  # medium permeability field (m^2)
# parameters["k_z_bed"] = 5E-10  # medium permeability field (m^2)
# parameters["S_e_bed"] = 4.1E-14  # specific storage (m.min^2/kg)
