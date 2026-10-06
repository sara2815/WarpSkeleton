
## Overview

GPU-accelerated skeletal animation with Forward Kinematics (FK), Inverse Kinematics (IK), and Linear Blend Skinning (LBS), built on [NVIDIA Warp](https://github.com/NVIDIA/warp) with a [Polyscope](https://polyscope.run/) UI.

Entry point: `src/main.py`

## Setup

```bash
pip install warp-lang polyscope scipy numpy
```

(or from the same env in A2)

```bash
conda activate cpsc426
conda install scipy -y
```

## Running

```bash
python src/main.py              # uses preferred device (GPU if available)
python src/main.py --device cpu # force CPU execution
python src/main.py --IK         # enable IK for Part III
```

## Project Structure

```
src/
  main.py               — UI + orchestration
  cli.py                — CLI 
  utils.py              — Data loaders
  skeleton.py           — Skeleton data container 
  mesh.py               — Mesh data container
  forward_kinematics.py — FK kernels
  skinning.py           — Skinning kernels 
  ik_solver.py          — IK kernels + solver loop 
data/
  skeleton_bind.json    — Bone hierarchy, bind-pose local matrices
  base_mesh.obj         — Character mesh (Z-up meters, converted to Y-up cm on load)
  vertex_weights.npz    — Per-bone vertex weights
  lod3.fbx              — Original full-body rig (reference, e.g. for Blender)
```

## What You Need to Implement
Recommended order: FK kernels -> Skinning kernels -> IK solver.