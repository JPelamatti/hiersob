## Sobol' indices in the presence of hierarchical variables
This repository contains the script allowing to compute first and total order Sobol' indices in the presence of hierarchical variables by relying on k-nearest neighbors estimation. It contains 3 files:
- Indices_computation.py, providing the estimation of the Sobol' indices based on an i.i.d. data set
- GSobol_Example.py, providing an example on the G-Sobol function in which the data is generated and the Sobol' indices are computed relying on the Sobol_computation.py script
- Reference_results.py, providing a script allowing to compute reference values for the Sobol' indices in the presence of hierarchical variables through crude Monte Carlo sampling

For more details on the underlying theory and the indices estimation, the reader is referred to the paper draft.
  
