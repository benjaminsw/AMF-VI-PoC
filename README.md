# AMF-VI Proof of Concept

A simplified implementation of Adaptive Mixture-Flow Variational Inference for testing the core concept.

## Features

- **Multiple Flow Types**: Real-NVP, Planar, and Radial flows
- **Automatic Mode Separation**: Regularization prevents mode collapse
- **IWAE Training**: Importance-weighted training for better mode capture
- **Comprehensive Evaluation**: Coverage, quality, and diversity metrics

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run basic demo:
```bash
bashpython examples/demo_2d_multimodal.py
```
3.Compare flow types:
```bash
bashpython examples/demo_comparison.py
```
4. Test on two moons:
```bash
bashpython examples/demo_moons.py
```