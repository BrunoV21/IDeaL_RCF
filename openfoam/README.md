# OpenFOAM v2006 integrations

## Installation and Set up
Follow this [guide](https://www.cemf.ir/how-to-install-openfoam-v2006-from-source-pack/) to install and configure OpenFOAM v2006 from the source pack.


## Custom Solvers
Once OpenFOAM is set up, an existing solver must be adapted to receive the predicted quantities and inject them into the momentum equations. The required changes are usually contained in the createFields.H and UEqn.H files.

A ML_pimpleFoam custom solver is provided which can be compiled by following the steps in this [video](https://www.youtube.com/watch?v=MiUDCOhbQaM).

Alternatively, the provided createFields.H and UEqn.H files can be used to customize an existent OpenFOAM solver.

## Converged RANS Simulations
The starting point for injection consists of a converged RANS simulation as provided by McConkey *et al.* in [A curated dataset for data-driven
turbulence modelling](https://doi.org/10.34740/kaggle/dsv/2637500).

## Injection
The resulting foam compatible fields generated via ideal_rcf should be placed in the last RANS state corresponding to the starting point of the injection simulation.

The changes made to the original runs provided by McConkey *et al.* for the PHLL cases are included in the ```PHLL_used_configs``` directory

## Post Processing
The resulting fields can be analyzed using the ```foam.postprocess``` and ```foam.visualization``` modules for plotting and evaluation of the ```ideal_rcf``` package.