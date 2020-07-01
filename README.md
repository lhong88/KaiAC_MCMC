
-----------
Description
-----------

This repository contains:

1. A package (`/Kai_MCMC`) for doing Bayesian Markov Chain Monte Carlo (MCMC) simulation on a kinetic model of the Kai oscillator.

2. An example MCMC run with input scripts and simulation outputs (`/example_run`). It is recommended to perform the simulations on a computing cluster. A Mathematica notebook for analysis is available in `/example_run/step5_sample`.

3. A phosphorylation dataset for model fitting (`/data`).

For a detailed description of the model, the fitting method, and the training dataset, see our preprint on [bioRxiv](https://www.biorxiv.org/content/10.1101/835280v1.abstract) or our paper in [_Molecular Systems Biology_](https://www.embopress.org/doi/full/10.15252/msb.20199355).

------------
Installation
------------

The folder containing the simulation package (`Kai_MCMC`) needs to be in the `PYTHONPATH` environment variable. On a Linux operating system, this is done by adding the following line to the `.bashrc` file:

```
export PYTHONPATH="$PYTHONPATH:path_containing_Kai_MCMC"
```

The simulation package depends on the following python modules: scipy, mpi4py, numba, [emcee](https://github.com/dfm/emcee), and [odespy](https://github.com/hplgit/odespy).

------------
License
------------

Copyright (c) 2020 Lu Hong, Danylo O. Lavrentovich, Eugene Leypunskiy, and Eileen Li.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
