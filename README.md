# OASIS : Optimal Arrangements for Sensing in SLAM

##Reference
kindly cite our paper if you find this library useful:
- Kaveti, P., Giamou, M., Singh, H., & Rosen, D. M. (2023). [**OASIS: Optimal Arrangements for Sensing in SLAM.**](https://arxiv.org/pdf/2309.10698.pdf). IEEE Intl. Conf. on Robotics and Automation (ICRA), 2024

 ```bibtex
@article{kaveti2023oasis,
  title={OASIS: Optimal Arrangements for Sensing in SLAM},
  author={Kaveti, Pushyami and Giamou, Matthew and Singh, Hanumant and Rosen, David M},
  journal={arXiv preprint arXiv:2309.10698},
  year={2023}
}
 ```
##Dependencies
Tested on ubuntu 20.04 LTS with Python3. You will need the following libraries.
- numpy
- scipy
- gtsam (https://github.com/borglab/gtsam)
- matplotlib

Current version of the code support only simulated experiments.
We are working on real dataset evaluation - stay tuned!

## Run Simulation Examples
- Use `--help` option to list all options to run the code. 
- To run optimization for average performance across multiple simulations
```bash
python3 main_expectation.py
```
- To run optimization for single simulation
```bash
python3 main.py
```
