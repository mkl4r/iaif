# Intermittent Active Inference

Repository accompanying the paper ["Intermittent Active Inference"](https://www.mdpi.com/1099-4300/28/3/269) submitted to the entropy special issue "Active Inference in Cognitive Neuroscience".
The repository contains the mouse model used in the paper ("mouse_simple.py"), code to run different Intermittent Active Inference agents which control the mouse cursor ("run_iaif_agents.py"), and a Jupyter notebook to generate visualisations.


## Installation

### Prerequisites
- Python ≥ 3.11
- JAX (CPU, GPU, or TPU support available)

### Setup
1. Clone the repository:
   ```bash
   git clone https://https://github.com/mkl4r/iaif.git
   cd iaif
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

   For GPU support:
   ```bash
   pip install -e .[gpu]
   ```

   For TPU support:
   ```bash
   pip install -e .[tpu]
   ```

   For CPU-only (default):
   ```bash
   pip install -e .[cpu]
   ```

## Repository Structure

### Core Components

- **`run_iaif_agents.py`**: Defines (Intermittent) Active Inference agents and runs simulations. Uses the AIF agent from [difai-base](https://github.com/mkl4r/difai-base) (our package for general Active Inference agents). Results are saved to `data/simulations/`
- **`mouse_simple.py`**: Contains the 1D mouse cursor model and plotting tools.
- **`visualise_results.ipynb`**: Jupyter notebook for generating plots similar to those in the paper.

### Data Structure

- **`data/simulations/`**: Simulation results and outputs
- **`data/plots/`**: Generated plot outputs

## Usage

### Running Simulations
To run simulations with (Intermittent) Active Inference agents:
```bash
python run_iaif_agents.py
```

### Creating Plots
Use the Jupyter notebook to generate visualizations:
```bash
jupyter notebook visualise_results.ipynb
```

This notebook creates plots similar to those presented in the paper.


## Troubleshooting

### CUDA Version Compatibility
The current setup uses JAX for CUDA 13 (keep in mind that GPU support is very limited on Windows/Mac). If you have CUDA 12.x installed, manually install the appropriate JAX version before installing this repository:

```bash
# For CUDA 12.x, install JAX manually first:
pip install --upgrade "jax[cuda12]" 

# Then install aif-pointing:
pip install -e .
```

Check the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for the correct JAX version for your CUDA setup.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@inproceedings{klar2026iaif,
  title={Intermittent Active Inference},
  author={Markus Klar and Sebastian Stein and Fraser Paterson and John H. Williamson and Henrik Gollee and Roderick Murray-Smith},
  booktitle={Active Inference in Cognitive Neuroscience},
  year={2026},
  url={}
}
```

## Contact

- **Author**: Markus Klar
- **Website**: [mkl4r.github.io](mkl4r.github.io)
- **Email**: markus.klar@glasgow.ac.uk