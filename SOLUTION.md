The report on the project SMILES-2026 Signal Interference Cancellation

# Abstract

In this project it is necessary to tackle the problem of the cleaning the signal from the interference. It is already presented a baseline solution which is included in this solution. Moreover, there is a function rank1_from_band_matrix in the code task_and_baseline.py which allows us to clean the signal from from the external interference. The external noise influences channels in the same way therefore we can say that this noise is coherent. So, rank1_from_band_matrix function isolates this specific external noise. In order to do it, this function reduces data dimensionality, as we become 4-demension sygnal (from 4 channels), meanwhile a coherent interference is 1-dimesional, as it has only one source. In other words, it calculates the covariance matrix and takes only the eigenvector with the largest eigenvalue.

Applying only the baseline the metric equals 4.02 dB. After the adding of rank1 to baseline the metric grows up to 7.01 dB that is lower than the necessary 8.0 dB. That’s why the additional methods were integrated that lead to the metric 9.15 dB.

There are also two principal restrictions in this task. Firstly, all modifications should be made inside the single function (your_canceller). Secondly, the work of scorer which estimates a result. The scorer breaks the removed component into three parts: 

tx_part = fit_tx_prediction(rx - rx_hat) # projects subtraction onto the strict 130-dimensional basis

residual = removed_band - # extracts the single strongest spatial component using eigenvalue decomposition (rank1_from_band_matrix)

err = residual - rank1_part # unexplained error

The first and second steps have to explain >= 0.95 of the noise.


# Reproducibility 

## 1.1 Environment

For this code you need the following libraries: Python (3.12.13), numpy (2.0.2), scipy (1.16.3). To install them run the following command:

pip install numpy scipy

To run the script use this command:

!python applicant_solution.py

## 1.2 Impementation details

After you run the code you get the results.json with the baseline and solution metrics. The data challenge_challenge.mat is provided 
in this project, as the downloading from google.disk is sometimes impossible due to frequent loading attempts.

# 2. Final solution description

## 2.1 Modifications

All modification were done only within the your_canceller function in the file applicant_solution.py. The rest of this file and the content of the task_and_solution.py remained unchanged.

**EXCEPT** the line 16 is a manual version of downloading the source-data if it is impossible to load it via a link https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing
The file with data challenge_challenge.mat is provided in this project.

## 2.2 Final approach

As it was already mentioned, baseline + rank1 solutions reach only the 7.01 dB, meanwhile in the task solution is required to be upper than 8.0 dB. Therefore, in addition to them, were also implemented iterative methods and simultaneous algebraic reconstruction technique (SART) displayed the max score - 9.15 dB.

  ch0: 10.19 dB
  ch1: 8.24 dB
  ch2: 10.42 dB
  ch3: 7.74 dB
  Metric: 9.15 dB

## 2.3 The choice of SART

The idea of using iterative methods was inspired by medical imaging which also tackles the problem of the interference in ECG, tomography and other types of imaging (1).
  The Simultaneous Algebraic Reconstruction Technique (SART) is an iterative solver belonging to the family of weighted gradient descent methods. It improves upon the classical Kaczmarz method (ART) by updating the solution vector simultaneously across all projections, thereby mitigating the localized limit-cycle artifacts typical of sequential block-iterative methods (2). 

## 2.4 The most significant contribution to the result

In the SART algorithm is possible to vary iterations and lambda (relaxation parameter). Mathematically, it serves as a scaled step size that controls the magnitude of the model update along the gradient of the weighted least-squares cost function. In the Table 1 you can see the variation of these parameters. 

**Table 1.** The variation of iteration and lambda parameters in SART algorithm. Th highest scores are marked with green.

| Lambda ($\lambda$) | Iteration = 1 | Iteration = 2 | Iteration = 3 |
| :--- | :---: | :---: | :---: |
| **1.0** | 8.02 dB | 8.72 dB | 9.07 dB |
| **1.1** | 8.11 dB | 8.82 dB | 9.13 dB |
| **1.2** | 8.20 dB | 8.91 dB | **9.15 dB** |
| **1.3** | 8.29 dB | 8.99 dB | **9.15 dB** |
| **1.4** | 8.37 dB | 9.05 dB | 9.13 dB |
| **1.5** | — | 9.10 dB | 9.09 dB |
| **1.6** | — | 9.13 dB | 9.03 dB |
| **1.7** | — | **9.15 dB** | 8.96 dB |
| **1.8** | — | **9.15 dB** | 8.88 dB |
| **1.9** | — | 9.14 dB | 8.78 dB |

# 3. Experiments and failed attempts

## 3.1 Random Forest

Random Forest model is used for interference suppression in some investigations (3). It is capable to detect non-linear relationships. The RF was implemented with parameters rf_args = {'n_estimators': 15, 'max_depth': 6, 'n_jobs': -1, 'random_state': 42}, but the metric reached only 6.35 dB. Possibly, the other combination of parameters could solve the problem, however maybe because of correlation between noise and target signal it can’t separate it properly.

=== Your Solution ===
  ch0: 6.89 dB
  ch1: 6.07 dB
  ch2: 7.54 dB
  ch3: 4.91 dB
  Metric [yours]: 6.35 dB

## 3.2 Landweber method

Landweber iteration belongs to iteration methods and was also successful displaying the result:

=== Your Solution ===
  ch0: 10.17 dB
  ch1: 8.26 dB
  ch2: 10.43 dB
  ch3: 7.67 dB
  Metric [yours]: 9.14 dB

With parameters iterations = 1, alpha = 0.5.

## 3.3 ART-metod

ART method also beats the result of 8 dB, however with lower values comparing with other iteration methods, including SART and Landweber iteration. The best result was reached with iterations = 2

  ch0: 9.69 dB
  ch1: 7.97 dB
  ch2: 10.22 dB
  ch3: 6.99 dB
  Metric : 8.72 dB

SART deals better with non-correlated errors in comparison to ART. 





1.	Borràs M, Chamorro-Servent J. Electrocardiographic Imaging: A Comparison of Iterative Solvers. Front Physiol. 2021 Feb 3;12. doi:10.3389/fphys.2021.620250 
2.	Andersen A. Simultaneous Algebraic Reconstruction Technique (SART): A superior implementation of the ART algorithm. Ultrason Imaging. 1984. doi:10.1016/0161-7346(84)90008-7 
3.	Liu J. Suppression of polarization random noise in a two-dimensional force sensorbased on random forest. J Sens Sens Syst. 2025 Jan 17;14(1):1–11. doi:10.5194/jsss-14-1-2025 
