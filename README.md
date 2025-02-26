# Code Implementation for [KDD2025 Submitted] ADFN : Automatic Differential Functional Network 


ADFN is a framework designed to transform black-box neural network models into transparent, white-box representations without compromising performance. This repository contains the implementation of the method described in the paper "Automatic Differentiable Functional Networks (ADFN)," which is currently under review at KDD 2025.

Repository and code being refactored and rewritten


---

## Abstract

> Despite their versatility, fully-connected neural networks remain uninterpretable. Current methods, such as feature analysis and modular decomposition, face challenges like human subjectivity, ambiguous outputs, and performance trade-offs. We introduce Automatic Differentiable Functional Networks (ADFN), a framework that automatically approximates complex neural networks through gradient-based optimized compositions of interpretable functions. Unlike existing approaches, ADFN does not rely on restrictive function libraries or require expert intervention to decompose networks into efficient, deterministic functional components, thereby serving as a viable post-hoc interpretability tool. Based on the established interpretability of target architectures, we conducted experiments on diverse architectures, including attention-based Transformers for modular addition and linear-based neural network models for Long-term Time Series Forecasting (LTSF), to demonstrate that ADFN approximates and interprets network behaviors while maintaining task performance. ADFN transforms internal modules of black-box models into transparent, efficient white-box representations without performance loss, advancing practical interpretable AI deployment.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)


## Features
