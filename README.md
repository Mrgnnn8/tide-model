# TiDE Model — PyTorch Forecasting

A PyTorch Forecasting implementation of the **TiDE (Time-series Dense Encoder)** model for long-term time series forecasting, based on the architecture proposed by Das et al. (2023).

## Overview

TiDE is an MLP-based encoder-decoder model that handles past time series data alongside covariates, achieving performance comparable to or better than Transformer-based models while being significantly faster.

This repository contains a patched version of the TiDE implementation from [pytorch-forecasting](https://github.com/sktime/pytorch-forecasting), fixing a bug where `self.future_cov` in `__init__` was incorrectly sized to include all known and unknown inputs, rather than only the dimensionality of the known covariates.

## Bug Fix

In the original `__init__`, `self.future_cov` was set to the size of all inputs (known + unknown covariates), which was incorrect. The fix correctly sets it to reflect only the known covariate dimensionality.

## Reference

> Das, A., et al. "Long-term Forecasting with TiDE: Time-series Dense Encoder." arXiv:2304.08424 (2023).
