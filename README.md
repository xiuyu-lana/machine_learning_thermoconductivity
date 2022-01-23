# Introduction
Finding the most printable alloys for metal additive manufacturing is an important and difficult task. Printability is strongly determined by the thermal conductivity of the alloy. However, the computational cost for calculating thermal conductivities accurately (usually done with molecular dynamics) is extremely high. Therefore, we used machine learning to greatly reduce this computational cost.

# Method
We use Weidmann-Franz law to get electronic contribution to thermal conductivity in each element of the alloy (`alpha_i`), where the chemical composition is `Xi`. We will apply a machine learning model to physical property data for individual elements in order to calculate a parameter `Omega_ij`, which in turn can the total electronic contribution to the conductivity of the alloy. The remaining phononic contribution will be calculated analytically in order to get the total thermal conductivity. The features of the model will be the alloy’s chemical composition and physical parameters (e.g. mechanical and electrical properties). We will test it first on some well-characterized alloys, such as steels like 316L, 904L, 304, Ni-Ti, and Inconel 718. The model can then be used to predict other alloys to find new and better printable alloys.

# Reference
- Gurunathan, Ramya, et al. "Analytical models of phonon–point-defect scattering." Physical Review Applied 13.3 (2020): 034011.
- J. Callaway, Model for lattice thermal conductivity at low temperatures, Phys. Rev. 113, 1046 (1959)
