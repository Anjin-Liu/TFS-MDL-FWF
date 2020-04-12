
## Source Code for MDL-FWF

### Title: Concept Drift Detection: Dealing with Missing Values via Fuzzy Distance Estimations
### Status: Under Review
### Abstract:
In data streams, the data distribution of arriving observations at different time points may change – a phenomenon called concept drift. While detecting concept drift is a relatively mature area of study, solutions to the uncertainty introduced by observations with missing values have only been studied in isolation. No one has yet explored whether or how these solutions might impact drift detection performance. Currently, the most elegant way to resolve missing values is data imputation. We, however, believe that data imputation methods may actually increase uncertainty rather than reducing it, and that imputation can introduce bias into the process of estimating distribution changes during drift detection. Moreover, we conjecture that appropriately considering missing values during drift detection, as opposed to imputation as an isolated preprocessing step, may improve accuracy beyond the current state-of-the-art. Our idea is to focus on estimating the distance between observations rather than estimating the missing values, and to define membership functions that allocate observations to histogram bins according to the estimation errors. Our solution comprises a novel masked distance learning (MDL) algorithm to reduce the cumulative errors caused by iteratively estimating each missing value in an observation and a fuzzy-weighted frequency (FWF) method for identifying discrepancies in the data distribution. Experiments on both synthetic and real-world data sets demonstrate the advantages of this method and show its robustness in detecting drift in data with missing values. Our results reveal that missing values exert a profound impact on concept drift detection, but using fuzzy set theory to model observations can produce more reliable results than imputation.



### Authors

* [**Anjin Liu**](https://www.uts.edu.au/staff/anjin.liu)
* [**Jie Lu**](https://www.uts.edu.au/staff/jie.lu)
* [**Guangquan Zhang**](https://www.uts.edu.au/staff/guangquan.zhang)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

The work presented in this paper was supported by the Australian Research Council (ARC) under Discovery Project DP190101733.