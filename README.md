# ADFN : Automatic Differential Functional Network 
## [KDD2025 Submitted, Under Review.] 
Code Implementation.


ADFN is a framework designed to transform black-box neural network models into transparent, white-box representations without compromising performance. This repository contains the implementation of the method described in the paper "Automatic Differentiable Functional Networks (ADFN)," which is currently under review at KDD 2025.

Repository and code being refactored and rewritten, current repository contains all the scripts for experiments.


---

## Abstract

> Despite their versatility, fully-connected neural networks remain uninterpretable. Current methods, such as feature analysis and modular decomposition, face challenges like human subjectivity, ambiguous outputs, and performance trade-offs. We introduce Automatic Differentiable Functional Networks (ADFN), a framework that automatically approximates complex neural networks through gradient-based optimized compositions of interpretable functions. Unlike existing approaches, ADFN does not rely on restrictive function libraries or require expert intervention to decompose networks into efficient, deterministic functional components, thereby serving as a viable post-hoc interpretability tool. Based on the established interpretability of target architectures, we conducted experiments on diverse architectures, including attention-based Transformers for modular addition and linear-based neural network models for Long-term Time Series Forecasting (LTSF), to demonstrate that ADFN approximates and interprets network behaviors while maintaining task performance. ADFN transforms internal modules of black-box models into transparent, efficient white-box representations without performance loss, advancing practical interpretable AI deployment.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)


## Experiments

This repository includes three distinct experiments to demonstrate the versatility and effectiveness of ADFN:

- **Toy Example:**  
  Located in the `adfn` folder, this experiment provides a simple, illustrative implementation of ADFN. It serves as an entry point for understanding the core concepts and functionality of the framework.

  ![mixed_functions (1)-1](https://github.com/user-attachments/assets/01f0fd52-f1e4-452e-b920-631f59f078c2)


- **Long-term Time Series Forecasting (LTSF):**  
  Found in the `adfn-ltsf` folder, this experiment applies ADFN to a linear-based neural network model designed for LTSF. It showcases the framework’s capability to maintain performance in forecasting tasks while offering interpretability.
![ETTh1_Electricity_pushed_ex-1](https://github.com/user-attachments/assets/db7fb0ff-3603-4ab7-806d-e1f1d5d57d73)

- **Transformers (ADFN-Grok):**  
  The `adfn-grok` folder contains experiments with attention-based Transformer models. This setup demonstrates the use of ADFN in decomposing and interpreting complex Transformer architectures, particularly in tasks like modular addition.
![grok_loss_plots-1](https://github.com/user-attachments/assets/a5783854-887d-4b2a-9374-4f522e54b7d6)

---

## Repository Structure

- **`adfn/`**: Contains the toy example implementation.
- **`adfn-ltsf/`**: Contains the implementation for Long-term Time Series Forecasting.
- **`adfn-grok/`**: Contains the implementation for Transformer-based experiments.

---

/## Getting Started

/To get started with ADFN, clone the repository first,
/enter the corresponding folder;

/```
/#case 1 : ADFN (toy)
/```

/```
/#case 2 : ADFN - LTSF
/```

/```
/#case 3 : ADFN - Grok 
/```
