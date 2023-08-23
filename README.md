# percussion
Likelihood models for HIV epidemiological data

# Introduction
HIV epidemic models are calibrated to various kinds of data, including HIV prevalence measured in population-based surveys like the Demographic and Health Surveys, or in sentinel population studies; case surveillance and vital registration data on new HIV diagnoses and HIV-related deaths; or HIV incidence and coverage of antiretroviral therapy measured in recent Population-based HIV Impact Assessment (PHIA) surveys. This package provides Python implementations of likelihood models that may be used to evaluate  goodness-of-fit of HIV epidemic models to survey and surveillance data.

This package currently implements likelihood models for HIV prevalence data only, but we intend to extend this to include the various data used to calibrate models in the UNAIDS-supported [Spectrum software suite](https://avenirhealth.org/software-spectrum.php).

The 'percussion' package is aspirational - a longstanding goal of the [UNAIDS Reference Group on Estimates, Modelling, and Projections](https://epidem.org/) has been to consolidate the [models and tools](https://hivtools.unaids.org/) used to make HIV estimates into one "symphony" model.

# Modules
## ancprev
Likelihood model for HIV prevalence measured among women attending antenatal care (ANC)

HIV prevalence data at ANC consists of sentinel surveillance done at selected ANC sites, routine HIV testing in pregnant women at those same sites, and  "census"-level routine testing of pregnant women aggregated to regional or national level. The ANC likelihood models are complex to account for numerous potential sources of bias, including systematic differences in HIV prevalence between pregnant women and adults overall, non-random selection of surveillance sites, and other factors like disruptions to test kit supply.

The likelihood model for ANC data has been used in the Estimates and Projection Package (EPP) years, with elaboration as different kinds of ANC data became availabl, and to account for different sources of bias as evidence for their effects accumulated. For technical details, see the following references:
- Alkema L, Raftery AE, Clark SJ. Probabilistic projections of HIV prevalence using Bayesian melding. Ann Appl Stat. 2007;1(1):229-248. [doi:10.1214/07-aoas111](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-1/issue-1/Probabilistic-projections-of-HIV-prevalence-using-Bayesian-melding/10.1214/07-AOAS111.full)
- Brown T, Bao L, Eaton JW, et al. Improvements in prevalence trend fitting and incidence estimation in EPP 2013. AIDS. 2014;28(Suppl 4):S415-S425. [doi:10.1097/QAD.0000000000000454](https://journals.lww.com/aidsonline/fulltext/2014/11004/improvements_in_prevalence_trend_fitting_and.2.aspx)
- Eaton JW, Bao L. Accounting for nonsampling error in estimates of HIV epidemic trends from antenatal clinic sentinel surveillance. AIDS. 2017;31(Suppl 1):S61-S68. [doi:10.1097/QAD.0000000000001419](https://journals.lww.com/aidsonline/fulltext/2017/04001/accounting_for_nonsampling_error_in_estimates_of.8.aspx)
- Sheng B, Marsh K, Slavkovic AB, Gregson S, Eaton JW, Bao L. Statistical models for incorporating data from routine HIV testing of pregnant women at antenatal clinics into HIV/AIDS epidemic estimates. AIDS. 2017;31(Suppl 1):S87-S94. [doi:10.1097/QAD.0000000000001428](https://journals.lww.com/aidsonline/fulltext/2017/04001/statistical_models_for_incorporating_data_from.11.aspx)

The likelihood model implementation in this package is based on Jeff Eaton's R and C implementations for [site-level data](https://github.com/jeffeaton/anclik) and for [census-level data](https://github.com/mrc-ide/eppasm).

## hivprev
Likelihood models for HIV prevalence data from population-based surveys

Spectrum models represent the likelihood of HIV prevalence data from household surveys in multiple ways. These methods start by assuming that a binomial model describes the number of people who test positive for HIV among survey respondents tested for HIV. However, population-based surveys have complex designs, and survey respondent data are typically weighted to improve the representativeness of the survey estimates. This weighting can mean that the weighted number of people tested is usually not a whole number, which prevents us from using a binomial model directly. Spectrum models handle this by generalizing the discrete binomial model to continuous data through different approximations.

The hivprev model assumes instead that the observed prevalence follows a Beta distribution given the (weighted) number of people tested for HIV. Specifically, given observed prevalence $P$ measured among $N$ respondents, the likelihood of the observed data given modeled HIV prevalence $\theta$ is derived as

$$P\sim \mathrm{Beta}(\theta N, (1-\theta)N)$$

# How to use the percussion package

You can install the `percussion` package from github using `pip`

```
python pip -m install git+https://${user}:${token}@github.com/rlglaubius/percussion.git
```

Here, `${user}` is your github username and `${token}` is your github-generated [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic).

## general
All likelihood values returned by percussion modules are on the (natural) log scale rather than the natural scale.

## ancprev
To use the `ancprev` module, you need a CSV file with your ANC data and a time series of HIV prevalence estimates in pregnant women, as in the Python code below:

```
import numpy as np
from percussion import ancprev

pwprev = np.array([0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00006, 0.00016, 0.00030, 0.00055,
                   0.00099, 0.00175, 0.00302, 0.00510, 0.00843, 0.01354, 0.02089, 0.03084, 0.04351, 0.05842,
                   0.07439, 0.08936, 0.10141, 0.11015, 0.11538, 0.11711, 0.11577, 0.11198, 0.10660, 0.10036,
                   0.09369, 0.08681, 0.07997, 0.07347, 0.06773, 0.06277, 0.05856, 0.05504, 0.05194, 0.04944,
                   0.04741, 0.04573, 0.04423, 0.04253, 0.04105, 0.03985, 0.03811, 0.03614, 0.03434, 0.03232,
                   0.03021, 0.02871, 0.02751, 0.02587, 0.02393, 0.02205, 0.02033, 0.01876, 0.01736, 0.01610,
                   0.01496])
anc = ancprev.ancprev(1970)
anc.read_csv("tests/ken-coast-anc.csv")
print(anc.likelihood(pwprev))
```

The pregnant women prevalence trend `pwprev` is the HIV epidemic model output we want to use to compare to the data. This includes one prevalence estimate per year. The snippet constructs an `ancprev` instance named `anc` using the constructor `ancprev(year)`, where `year` is the year the first value in `pwprev` pertains to. We initialize `anc` with our ANC data from a CSV file, `ken-coast-anc.csv`, which can be found in the `tests` folder of this repository. The last line of code evaluates the likelihood of seeing this ANC data if `pwprev` were the true HIV prevalence among pregnant women.

The ANC data file format is a plain-text CSV file with one prevalence datapoint per row, and the following columns:
- `Region`: The geographic region (e.g., a province name, urban/rural setting, etc). This is used to distinguish different ANC sites that have the same name.
- `Site`: The facility name, or `Census` for regional or national aggregate data.
- `Type`: `SS` for sentinel surveillance data or `RT` for routine testing data.
- `Year`: The year the datapoint was measured.
- `Prevalence`: The measured HIV prevalence. This must be a proportion between 0 and 1.
- `N`: The number of women whose HIV status was ascertained. For sentinel surveillance, this is the number of women tested. For routine testing, this is the number tested at ANC plus the number whose HIV status was known at the first ANC visit of their pregnancy.
- `UseDataInFit`: Datapoints are included in likelihood evaluation if this is `TRUE` and are excluded if this is `FALSE`. This can be used to exclude data from evaluation, e.g., in the case of data quality challenges.

The ordering of rows and columns does not matter. You may include additional columns with (e.g.) documentation for datapoints. These must not use any of the column names listed above. Any additional columns will be ignored when you initialize your `ancprev` instance.

## hivprev
To use the `hivprev` module, you first need a file with your survey-based prevalence data. See `mwi-hiv-prev-data.csv` in the `tests` folder of this repository for an example. This is a plain-text CSV file with one prevalence datapoint per row and the following columns:

- `Population`: This is a free-form string. The example file has data for female sex workers ("FSW"), men who have sex with men ("MSM") and general population ("All"). Your HIV epidemic model must be capable of supplying HIV prevalence estimates for each population in your CSV file.
- `Gender`: This is free-form. The example file has data for females, males, and both together ("All"). Your HIV epidemic model must be capable of supplying HIV prevalence estimates for each sex or gender in your CSV file.
- `AgeMin`: The youngest age of individuals contributing to the datapoint.
- `AgeMax`: The oldest age of individuals contributing to the datapoint.
- `Value`: The measured HIV prevalence. This must be a proportion between 0 and 1.
- `NumTested`: The number of people tested to produce the prevalence estimate. For complex surveys, this should be weighted to reflect the effective sample size, and need not be a whole number.
- `Year`: The year of the prevalence estimate.
- `UseDataInFit`: Datapoints are included in likelihood evaluation if this is `TRUE` and are excluded if this is `FALSE`.

The ordering of rows and columns does not matter. You may include additional columns with (e.g.) documentation for datapoints. These must not use any of the column names listed above. For example, the `mwi-hiv-prev-data.csv` has additional columns to record the indicator ("HIV prevalence"), survey source ("Source"), and notes on where the estimates were obtained. These additional columns will be ignored when you initialize your `hivprev` instance. 

When you decide what datapoints to include or exclude, note that the likelihood treats all datapoints as independent observations, which is not appropriate for handling overlapping oservations from the same survey. For example, if you have estimates for age 15-24, 25-49, and 15-49 from the same survey, then either the aggregate 15-49 estimate or the constituent 15-24 and 25-49 estimates should be excluded, otherwise the likelihood will give undue weight to these overlapping observations. This undue weight may bias epidemic model estimates calibrations informed by the likelihood.

The survey data file format is flexible in terms of populations and age groups you can represent. This flexibility does mean you are responsible for aligning your HIV epidemic model estimates with the different population groups you have survey data for. To help with this alignment, `hivprev` instances can construct a prevalence template that is sufficient to evaluate the likelihood for each datapoint included in your fit. This is shown in the Python code below:

```
from percussion import hivprev

hiv = hivprev.hivprev(1970)
hiv.read_csv("tests/mwi-hiv-prev-data.csv")
template = hiv.projection_template()
template['Prevalence'] = 0.1
print(hiv.likelihood(template))
```

We initialize the `hivprev` instance named `hiv` with the first year of our HIV epidemic model projection (1970) for consistency with other likelihood modules. After initializing `hiv` using `read_csv()`, we can request a projection template via `hiv.projection_template()`. This template is a pandas data frame with columns for `Population`, `Gender`, `AgeMin`, `AgeMax`, and `Year`, with one row for each unique combination of column values in the input CSV file. The `Prevalence` column in the template is initialized to `NaN`. You must fill in these prevalence values with prevalence (as a proportion) before evaluating the likelihood. In the example above, we simply set these values to 10% (0.1) for the sake of illustration.