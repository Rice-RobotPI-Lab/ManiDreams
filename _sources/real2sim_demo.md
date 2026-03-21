# Zero-shot Real-to-Sim Demo

Real-time perception-to-simulation: detect a real object via stereo camera, build a domain-randomized Newton physics simulation, and interact via a browser-based UI.

## Overview

This demo combines:
- **Intel RealSense D415** — stereo IR + RGB capture
- **Fast-FoundationStereo (FFS)** — zero-shot stereo matching → depth → point cloud
- **SAM2** — real-time object segmentation and tracking
- **Newton XPBD** — multi-world physics simulation with domain randomization
- **Viser** — 3D visualization with interactive gizmo control

The workflow:
1. **Select table** — SAM2 segments table surface, plane is fitted and locked
2. **Select object** — SAM2 tracks object, oriented bounding box (OBB) is estimated
3. **Simulate** — OBB parameters are used to build 32 domain-randomized Newton worlds
4. **Interact** — drag the gizmo in the Viser 3D view to push objects

## Prerequisites

**Hardware:**
- Intel RealSense D415 camera (USB 3.0)
- NVIDIA GPU with CUDA support (tested on RTX 3070+)

**Software:**
- Linux (tested on Ubuntu 22.04/24.04)
- CUDA 12.4+

## Installation

All steps assume you start from a clean conda environment.

### Step 1: Create environment and install PyTorch

```bash
conda create -n manidreams python=3.12 && conda activate manidreams

pip install torch==2.6.0 torchvision==0.21.0 xformers \
    --index-url https://download.pytorch.org/whl/cu124
```

### Step 2: Install ManiDreams

```bash
git clone https://github.com/Rice-RobotPI-Lab/ManiDreams.git
cd ManiDreams
pip install -e .
cd ..
```

### Step 3: Install Fast-FoundationStereoPose (includes SAM2)

```bash
git clone https://github.com/Vector-Wangel/Fast-FoundationStereoPose.git
cd Fast-FoundationStereoPose
pip install -r requirements.txt
pip install pyrealsense2
cd ..
```

```{note}
FFS depends on `opencv-contrib-python` while ManiDreams pulls `opencv-python` via mani-skill. Installing FFS requirements **after** ManiDreams ensures `opencv-contrib-python` takes precedence (it is a superset of `opencv-python`).
```

### Step 4: Download model weights

**FFS stereo model:**

Download from the link in the [FFS README](https://github.com/Vector-Wangel/Fast-FoundationStereoPose#readme) and place at:
```
Fast-FoundationStereoPose/weights/23-36-37/model_best_bp2_serialize.pth
```

**SAM2 checkpoint:**
```bash
mkdir -p Fast-FoundationStereoPose/SAM2_streaming/checkpoints/sam2.1
wget -P Fast-FoundationStereoPose/SAM2_streaming/checkpoints/sam2.1 \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### Step 5: Install Newton physics + visualization

```bash
pip install newton-physics warp-lang viser flask
```

## Running the Demo

The demo expects `Fast-FoundationStereoPose` as a sibling directory to `ManiDreams`:

```
your_workspace/
├── ManiDreams/
└── Fast-FoundationStereoPose/
```

```bash
cd ManiDreams
python examples/tasks/zeroshot_real2sim_demo/main.py
```

To specify a custom FFS location:

```bash
FFS_DIR=/path/to/Fast-FoundationStereoPose \
    python examples/tasks/zeroshot_real2sim_demo/main.py
```

The demo prints two URLs:
- **Web UI** — `http://localhost:909x` — left panel (RGB + SAM2 mask) + right panel (Viser 3D)
- **Viser** — `http://localhost:909y` — standalone 3D viewer

## Usage Guide

### Phase 1: Table Detection

1. Click **"1. Select Table"** in the toolbar
2. Draw a bounding box over the table surface (or click a point on it)
3. SAM2 segments the table, a plane is fitted over ~10 frames
4. Once variance converges, the plane **locks** automatically (shown as a blue quad in 3D view)

### Phase 2: Object Tracking

1. Click **"2. Select Point"** or **"2. Select BBox"**
2. Click on or draw a box around the target object
3. SAM2 tracks the object, an oriented bounding box (OBB) is estimated with temporal smoothing
4. The green wireframe OBB appears in the 3D view

### Simulate

1. Click **"Simulate"** — builds 32 Newton worlds with domain-randomized copies of the detected object
2. Perception pauses to free GPU VRAM for physics
3. Semi-transparent colored boxes appear in the 3D view
4. **Drag the gizmo** (RGB arrows) to push the actor into objects
5. Click **"Pause Sim"** to resume perception while keeping the simulation state

### Reset

Click **"Reset All"** to clear everything and start over with a new object.

## Architecture

This demo maps to ManiDreams components as follows:

| Demo Module | ManiDreams Component | File |
|---|---|---|
| FFS + SAM2 + OBB estimation | `D415FFSExecutor` (ExecutorBase) | `examples/executors/d415_ffs_executor.py` |
| Newton multi-world simulation | `NewtonBackend` (SimulationBackend) | `examples/physics/newton_backend.py` |
| Domain randomization | `randomize_envs()` in NewtonBackend | `examples/physics/newton_backend.py` |
| Viser gizmo → actor target | Action (no solver — teleop) | `examples/tasks/zeroshot_real2sim_demo/main.py` |
| Flask Web UI | Embedded in main script | `examples/tasks/zeroshot_real2sim_demo/main.py` |

The demo uses **no Solver or Cage** — it is a teleop demo. To add autonomous planning, connect a Solver + Cage to the existing TSIP, as shown in the [object pushing example](runnable_examples.md#object-pushing-simulation).
