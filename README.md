<h2 align="center">
Automated Machine Learning for Residential Property Valuation
</h2>

## Abstract

Many studies have shown that machine learning models outperform other approaches in residential property valuation. However, developing well-performing machine learning models requires the deep involvement of domain experts and data scientists, posing challenges for scale artificial intelligence (AI) implementation and application. Automated machine learning (AutoML) implements the end-to-end process of repetitive machine learning (ML) tasks without much need for human assistance. Although existing techniques have yielded promising results, no domain-specific automated machine learning framework exists for residential property valuation. Most of them are domain-agnostic and ignore domain knowledge, limiting their practical AI application in real-world scenarios. This study proposes a domain-specific automated machine learning framework for residential property valuation (AutoML4RPV), incorporating domain-specific (data manipulation and feature engineering) and domain-agnostic (machine learning pipeline generation) function modules to streamline AI implementation. Three real housing transaction datasets (New York, London, and Singapore), as well as a public valuation dataset are used for experimental validation by comparing AutoML4RPV with existing AutoML frameworks. Results show that AutoML4RPV achieved R squared of 58.4% for New York dataset, 25.6% for London dataset, 96% for Singapore dataset, and 82.5% for public valuation dataset. The raw data processed by the domain-specific modules of AutoML4RPV leads to larger R squared values than that evaluated by two domain-agnostic AutoML frameworks supporting raw data input, outperforming second-ranked framework by 17.6% for New York dataset, 18.2% for London dataset, and 9.80% for Singapore dataset. Using the finalized dataset delivered by AutoML4RPV, the domain-agnostic modules of AutoML4RPV achieves the best model performance for Singapore and public valuation datasets. A web app has been developed to facilitate domain experts and data scientists in building, comparing and deploying machine learning models from scratch. The system architecture of AutoML4RPV’s ultimate version is proposed to pave the way for future research.

## Research Methodology
![image](https://github.com/user-attachments/assets/2a453b08-4562-46c7-aed3-71bb788820ef)

Use following command to install necessary packages:

```
pip install requirements.txt
```
Run the python file `app.py` to initiate the web app via

```
http://127.0.0.1:8000/
```



