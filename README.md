[![DOI](https://zenodo.org/badge/731216583.svg)](https://zenodo.org/doi/10.5281/zenodo.10472002)
# Getting started

Here you find the code described in [borges2024caviar] (see below), which allows to generate the results in this paper.

\*Currently tested only on Ubuntu 22.04 and Python 3.9.16

> Recommended mixed-OS setup: run the AirSim Unreal executable manually on **Windows** and run the remaining Python/NATS pipeline on **WSL2 Ubuntu 20.04**.

## Pre-requisites

### Auxiliary linux packages

#### First update (if necessary)

    sudo apt update

#### cURL

    sudo apt install curl

#### Unzip

    sudo apt install unzip

### Setting up the NATS server

Go to https://github.com/nats-io/nats-server/tags , download the latest `.deb` release and install it.

## Installing

### 1) Clone the project repository

#### Using SSH:

    git clone git@github.com:lasseufpa/caviar.git

#### Using HTTP:

    git clone https://github.com/lasseufpa/caviar.git

### 2) Set up and start the 3D scenario (Windows host)

For the WSL2 workflow, keep the 3D executable on Windows and launch it manually before running `simulate.py` in WSL2.

1. Download/extract the `central_park` executable on Windows (same package used in this repository).
2. Ensure your AirSim `settings.json` on Windows includes your `Vehicles` entries (you can reuse the repository `settings.json`).
3. Start the Unreal executable manually on Windows and wait until the map is fully loaded.

> In this mode, `simulate.py` does **not** auto-start AirSim.

### 3) Install the requirements

We use python 3.9.16\*

(Optional): Create and activate a virtual environment with conda

```
conda create --name caviar python=3.9.16
```

```
conda activate caviar
```

First we should install the package below

```
pip install msgpack-rpc-python==0.4.1
```

After this, we install the rest of the requirements

```
pip install -r requirements.txt
```

## Executing a simulation

In WSL2 (Ubuntu 20.04), in the project root folder run:

    python3 simulate.py

This starts NATS + mobility + Sionna in WSL2 and connects to AirSim running on Windows. The flight paths are defined in `caviar/examples/airsimTools/waypoints/trajectories/`.

To correctly abort a simulation, in the terminal press:

    ctrl+C

## Configuring the simulation

The configuration parameters used in the simulation are stored in `caviar/caviar_config.py`

**More documentation on the configuration soon**

## Troubleshoot

#### On the first run, the drone was teleported to the street, but did not started to fly

Sometimes this error can happen due to the download of YOLO weights. For this case, just exit the current simulation with `ctrl+c` and try it again.

#### Got the error: "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may result in compilation or runtime failures, if the program we try to run uses routines from libdevice."

This is due to not being able to file the CUDA directory. For this you can execute the following, to add it to your environment variables:

##### For bash

    echo -e "export CUDA_DIR=\"$(whereis cuda | cut -d ' ' -f 2)\"\nexport XLA_FLAGS=--xla_gpu_cuda_data_dir=\"\${CUDA_DIR}\"" >> ~/.bashrc

##### For zsh

    echo -e "export CUDA_DIR=\"$(whereis cuda | cut -d ' ' -f 2)\"\nexport XLA_FLAGS=--xla_gpu_cuda_data_dir=\"\${CUDA_DIR}\"" >> ~/.zshrc

## Citation

If you benefit from this work, please cite on your publications using:

```
@ARTICLE{borges2024caviar,
  author={Borges, João and Bastos, Felipe and Correa, Ilan and Batista, Pedro and Klautau, Aldebaro},
  journal={IEEE Internet of Things Journal},
  title={{CAVIAR: Co-Simulation of 6G Communications, 3-D Scenarios, and AI for Digital Twins}},
  year={2024},
  volume={11},
  number={19},
  pages={31287-31300},
  doi={10.1109/JIOT.2024.3418675}}
```


## WSL2 ↔ Windows AirSim connection

The connection is configured in `caviar_config.py`:

- `airsim_host = "auto"` (default): resolves the Windows host IP from `/etc/resolv.conf` in WSL2.
- `airsim_port = 41451`: AirSim RPC port.
- `start_airsim_from_simulate = False`: keeps AirSim manual on Windows.

If needed, set `airsim_host` to a fixed Windows IP address.
