# DiffGLE: Differentiable Coarse-Grained Dynamics

## Overview
This repository **WILL** contain the implementation of **DiffGLE**, a differentiable coarse-grained dynamics framework based on the **Generalized Langevin Equation (GLE)**. The methodology leverages **Automatic Differentiation (AD)** and the **adjoint-state method** to accurately parameterize non-Markovian GLE models for coarse-grained fluids.

The code in this repository corresponds to our paper:

📄 **[DiffGLE: Differentiable Coarse-Grained Dynamics using Generalized Langevin Equation](https://arxiv.org/abs/2410.08424)**  
👨‍🔬 *Authors: Jinu Jeong, Ishan Nadkarni, Narayana R. Aluru*  

## Features
- **End-to-End Differentiable Simulation**: Implements a differentiable molecular dynamics (CGMD) framework.
- **Generalized Langevin Equation (GLE) Parameterization**: Uses a colored noise ansatz to parameterize memory kernels.
- **Adjoint-state methods for Optimization**: Efficiently optimizes CG models with memory kernel and colored thermal noise.
- **Validation on Complex Fluids**: Demonstrated on H2O, CO2, CH4, and Confined H2O.

## Requirements

## How to use

