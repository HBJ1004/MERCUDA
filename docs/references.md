# References

[MERCUDA overview](../README.md) · [Usage guide](../README_MERCUDA.md)

Cite Chambers (1999) for calculations based on MERCURY6, and record the MERCUDA
revision and force settings. Source comments identify the implemented models.

- Chambers, J. E. (1999), [A hybrid symplectic integrator that permits close encounters between massive bodies](https://doi.org/10.1046/j.1365-8711.1999.02379.x), *MNRAS* **304**, 793–799. MERCURY6 and its hybrid method.
- Will, C. M. (2014), [The Confrontation between General Relativity and Experiment](https://doi.org/10.12942/lrr-2014-4), *Living Reviews in Relativity* **17**, 4, equation 79. Implemented 1PN equation in the central-mass test-body limit (eta=0), with G and c restored.
- Tamayo, D., Rein, H., Shi, P. & Hernandez, D. M. (2020), [REBOUNDx](https://doi.org/10.1093/mnras/stz2870), *MNRAS* **491**, 2885–2901, Appendix B. Central-mass 1PN approximation and more complete alternatives.
- Burns, J. A., Lamy, P. L. & Soter, S. (1979), [Radiation forces on small particles in the solar system](https://doi.org/10.1016/0019-1035(79)90050-2), *Icarus* **40**, 1–48. Radiation pressure and PR physical background. MERCUDA's PR routine uses a component-wise transverse velocity that differs from the standard vector formula.
- Liou, J.-C., Zook, H. A. & Jackson, A. A. (1995), [Radiation Pressure, Poynting-Robertson Drag, and Solar Wind Drag in the Restricted Three-Body Problem](https://doi.org/10.1006/icar.1995.1120), *Icarus* **116**, 186–201. Radiation pressure, PR and solar-wind drag background for MERCUDA's PR routine.
- Klačka, J., Petržala, J., Pástor, P. & Kómar, L. (2012), [Solar wind and the motion of dust grains](https://doi.org/10.1111/j.1365-2966.2012.20321.x), *MNRAS* **421**, 943–959. Solar-wind force background for MERCUDA's PR routine, which uses a fixed solar-wind factor (0.3) rather than the full model in this paper.
- Farnocchia, D. et al. (2013), [Near Earth Asteroids with measurable Yarkovsky effect](https://doi.org/10.1016/j.icarus.2013.02.004), *Icarus* **224**, 1–13. Empirical transverse acceleration, here with distance exponent 2. The paper's Yarkovsky A2 coefficient is named `yar` in MERCUDA inputs.
- Marsden, B. G., Sekanina, Z. & Yeomans, D. K. (1973), [Comets and nongravitational forces. V](https://doi.org/10.1086/111402), *AJ* **78**, 211–225. Original A1/A2/A3 cometary distance law retained in MERCUDA.
- Murray, C. D. & Dermott, S. F. (1999), [Solar System Dynamics](https://doi.org/10.1017/CBO9781139174817), Cambridge University Press. Newtonian planetary dynamics and central-body zonal harmonics J2/J4/J6.
- Rein, H. & Spiegel, D. S. (2015), [IAS15: a fast, adaptive, high-order integrator for gravitational dynamics](https://doi.org/10.1093/mnras/stu2164), *MNRAS* **446**, 1424–1437. Independent stepping/reference integrator used in the accuracy benchmarks.
