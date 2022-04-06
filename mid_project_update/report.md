---
title: 6.435 Progress report - Boosting Variational Inference
output:
  pdf_document:
    latex_engine: xelatex
geometry: margin=2.5cm
fontsize: 11pt
font: Bookman
author:
  - Geoffrey Liu
keywords: 
---

## Project Overview

In this project, I explore different universal boosting 

Mean field variational inference is a widely used approach in Bayesian inference, where the posterior distribution is approximated with a variational distribution that factorizes across the parameters of the model. However, this assumption restricts the approximating distribution to be a unimodal distribution, and therefore cannot capture the multimodality. Moreover, MFVI tends to underestimate the posterior covariance, which can lead to false certainty about the parameter estimates.

A natural idea is to use a mixture distribution as the approximating variational distribution. Using simple components such as Gaussian's, we are able to approximate any continuous probability density arbitrarily well. 

## Proof

## What have you done so far? Report any preliminary findings (positive or negative)

From my original proposal, I have 

1. (100%) Implement Automatic Differentiation Variational Inference (ADVI) and Black Box Variational Inference (BBVI).

>I have implemented BBVI with mean-field Gaussian families and variational inference from scratch in JAX. I used the re-paramaterization trick for mean-field Gaussian families. I believe adding more families and full-rank is redundant, as I can resort to NumPyro which is also a Jax project.

2. (100%) For both debugging and educational purposes, create a visualization program to show refinement of ADVI and BBVI to the posterior using simple 1-D and 2-D distributions.

>I have created some simple visualizations to visualize that my BBVI on simple 1D and 2D distributions, such as a Gamma distribution, and a two dimensional Gaussian distribution with correlation.

1. (80%) Implement Universal BVI (UBVI) from Campbell et al. (2019) and add to visualization program.
  
>I have completed a literature review for UBVI and explored the literature of variational inference that do not use the KL-divergence as the variational objective. I have tested that Universal Boosting Variational Inference can be used to approximate a Cauchy

4. (Partially) Implement Black Box Boosting Variational Inference

> From the proposal phase, I found a ready to use implementation of Black Box Boosting Variational Inference on Pyro. I currently have this running for a toy example, however am failing to get this to converge for different examples e.g. Cauchy distribution

4. (04/13, pushed back) Implement BVI from Miller et al. (2016)



### Optional Steps

1. (Optional) Implement Stein Variational Gradient Descent (SVGD) and add to visualization program. Use ideas from BVI to formulate a boosting or sequential SVGD where particles are iteratively added.

I have re-prioritized this to be the first optional task, because after implementing boosting variational inference, SVGD seems very similar idea except using a mixture of dirac-delta distributions as the variational family.

2. (Optional) Add logistic regression experiments using the different implemented VI algorithms.



3. (pushed back) Implement BVI from Guo et al. (2016) and compare differences and analyze issues with joint weight and mixture optimization from step 3 using visualization program.

I have pushed this back from previous project and decided to focus only on BVI from Miller et al. (2016). The code-base is quite complex and is written in Julia, which will require me to learn a new coding language. This is not worth the extra cost since the idea is quite similar to BVI from Miller et al. (2016).







## What do you plan

There are different charts that I plan to recreate. The f

