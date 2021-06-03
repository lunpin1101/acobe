# Acobe

Acobe: An ***A***nomaly Detection Method Based on ***CO***mpound ***BE***havior

Lun-Pin Yuan, Euijin Choo, Ting Yu, Issa Khalil, and Sencun Zhu. 2021. Time-Window Based Group-Behavior Supported Method for Accurate Detection of Anomalous Users. In Proceedings of the 2021 51th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), June 21-24, 2021, Virtual Event, Taipei.

[https://arxiv.org/abs/2012.13971](https://arxiv.org/abs/2012.13971)

<img src="images/acobe_frontpage.jpg" width="512">

# System Requirement

This project was developed on ***BIZON G3000 – 2-4 GPU Deep Learning Workstation PC*** (spec listed below).  This project requires ***64 GB memory***, or it may crash during runtime.  It took *1-2 hours* to run a single executation (i.e., *exp-unswnb15.py* and *exp-sosp2009.py*).  A single execution includes training, testing, and generating evaluation metrics, but not preprocessing and plotting.

| Spec          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| Processor     | Skylake X; 8-Core 3.80 GHz Intel Core i7-9800X               |
| Memory        | DDR4 3000MHz 64 GB (4 x 16 GB)                               |
| Graphics Card | 2 x NVIDIA RTX 2080 8 GB with 1 x NVLink Bridge              |
| System        | Ubuntu 18.04 Bionic (not using Bizon's preinstalled package) |
| Environment   | Python 3.7.4, Tensorflow 2.0.0, and Anaconda 4.7.12          |

# Dataset

The ***data*** folder provides a few preprocessed data examples.  For full dataset, please find them from the links below. 

| Name      | Content | Link |
| --------- | ------- | ---- |
| CERT 2016 | Insider Threat | [https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099) <br> [https://doi.org/10.1184/R1/12841247.v1](https://doi.org/10.1184/R1/12841247.v1) |

Please preprocess the dataset by running the following command.

```
placeholder
```

# Execution Examples

Please use flags if needed.  For example, to specificy input directory, use *-i /data/cert2016/data*.  Please use *-h* for flag options.

```
placeholder
```

## Plot and Analysis

The ***paper*** folder provides the scripts we used when writing our paper.  The ***paper*** folder includes plotter scripts.  Please modify the directory paths before use.  

```
python3 paper/plot-cert2016.py
```



# Misc 

There are a few un-used code and un-used parameters.  They are our undergoing work.  I would suggest not to alter them, but feel free to explore.  Happy hacking!
