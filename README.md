# Orthogonal-by-construction model augmentation with JAX
Contains the code implementation and example scripts for the manuscript titled *Orthogonal-by-construction augmentation of physics-based input-output models* (currently submitted for publication). The preprint is available on [arXiv](https://arxiv.org/abs/2511.01321).

## Installation
To install the code implementation and the example scripts, first, clone the repository, then open the project folder as
```bash
git clone https://github.com/AIMotionLab-SZTAKI/orthogonal-IO-augm
cd orthogonal-IO-augm/
```

It is recommended to use a virtual environment, then install the package and its dependencies as
```bash
pip install -e .
```

For standalone installation (without the example scripts) use the following command
```bash
pip install git+https://github.com/AIMotionLab-SZTAKI/orthogonal-IO-augm@main
```

## Usage
To run the example script for the orthogonal-by-construction parametrization:
```bash
python3 examples/orth_by_constr_augm.py
```

Training options regarding the length of the estimation data, measurement noise, initialization, etc., can be specified by editing the corresponding parts of the script. For benchmarking, the standard additive model augmentation structure can be trained on the same example problem by running the script `examples/standard_additive_augm.py`. A physics-motivated simulation example can be found in `examples/NARX_example_orth_by_constr.py`. A comparison with regularization-based techniques is also available: `examples/NARX_example_regul_methods.py`.

## Citation
```
@article{gyorok_orthogonal_2025,
    title={Orthogonal-by-construction augmentation of physics-based input-output models}, 
    author={Bendegúz M. Györök and Maarten Schoukens and Tamás Péni and Roland Tóth},
    year={2025},
    journal={arXiv preprint arXiv:2511.01321}
}
```

## Funding
This project has received funding from the European Defence Fund programme under grant agreement number No 101103386 and has also been supported by the Air Force Office of Scientific Research under award number FA8655-23-1-7061. This work is also partly funded by the European Union (ERC, COMPLETE, 101075836). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Commission or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## License
See the [LICENSE](/LICENSE) file for license rights and limitations (MIT).
