import pandas as pd
import numpy as np
import scipy
from scipy import stats

from app import *
from typing import TypeVar, Generic, Union

from statsmodels.tsa.stattools import adfuller, kpss

__all__ = [
    "Statistics", "DescriptiveStatistic",
    "HypothesisTests", "Traditional", "Parametric",
    "NonParametric", "TimeSeries"
]


Arr = Union[pd.core.frame.DataFrame, pd.core.series.Series]

ArrLike = Union[
    pd.core.frame.DataFrame, 
    pd.core.series.Series,
    np.ndarray,
    list
]

class Statistics:

    def __init__(
        self,
        df: pd.core.frame.DataFrame
    ) -> None:
        self.df = df
        self.iqrs = self._make_iqrs()

    def _make_iqrs(
        self
    ) -> dict:
        keys = self.df.columns
        storage = {}
        #todo sensetive to nan -> add decorator to inform
        iqrs = stats.iqr(self.df, axis=0)

        for i in range(len(keys)):
            storage[keys[i]] = iqrs[i]
        return storage

    def FD_rule(
        self,
        column: str
    ) -> np.ndarray:
        '''
        #* Freedman_Diaconis rule can be 
        #* used to select the width of 
        #* the bins to be used in a histogram
        #* Parameters
        #* ----------
        #* column: str
        #* Series to apply FD_rule on
        #* Returns
        #* ----------
        #* np.ndarray
        '''

        iqr = self.iqrs.get(column)
        num_observation = len(self.df.loc[:, column])
        bin_width = 2 * iqr / np.cbrt(num_observation)

        return bin_width

    @staticmethod
    def filter_by_zscore(
        arr:  Arr,
        pvalue: float = 0.95,
        nan_policy="omit"
    ):
        '''
        #* Approach to detect and remove anomalies from dataset
        #* assuming dataset alike normal distribution
        #* zscores uses to standartize data by mean and stdev as follows:
        #* Z = (x - mean) / stdev
        #* the Z results can be interpret as:
        #* how far values lies beyond the mean
        #* to classify Z to normal / anomaly the are confidence intervals
        #* Example: pvalue = 0.95 -> critical value is +-1.96, so
        #* if Z-value beyond +-1.96 it's an anomaly
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''

        lower, upper = critical_values.get(pvalue)
        
        standartized = pd.Series(
            #todo sensetive to nan -> add decorator to inform
            stats.zscore(arr, nan_policy=nan_policy)
        )
        

        normal_index = standartized[
            (standartized < upper)
            & (standartized > lower)
        ].index

        #* drops rows where the value beyond the critical value 
        if type(pd.core.series.Series):
            new_arr = arr.loc[normal_index]
            
        else:
            new_arr = arr.loc[normal_index, :]
    
        return new_arr


class DescriptiveStatistic(Statistics):
    '''
    #* The aim is to provide some information
    #* about a given data
    #* Among existing methods some should be implemented:
    #*  Histplot, Boxplot, QQplot
    #*  mean, median, anomalies, std, 
    #*  skew, kurtosis, coeff_variation and so on
    #*
    #* Attributes
    #* ----------
    #*
    #* Methods
    #* ----------
    #*
    '''
    def __init__(self):
        return
    
class HypothesisTests:
    SIGNIFICANCE = 0.05
    NAN_POLICY="omit"

    HYPOTHESIS_DESC = {
        "normaltest":
            {
                "H0": "The data comes from a specified distribution",
                "H1": "The data does not come from a specified distribution"
            },

        "anderson":
            {
                "H0": "The data comes from a specified distribution",
                "H1": "The data does not come from a specified distribution"
            },
        "shapiro":
            {
                "H0": "The data comes from a specified distribution",
                "H1": "The data does not come from a specified distribution"
            },

        "kruskal":
            {
                "H0": "Population medians are equal",
                "H1": "Population medians are not equal"
            },
        "wilcoxon":
            {
                "H0": "Medians of two samples are equal",
                "H1": "Medians of two samples are different"
            },
        "mannwhitneyu":
            {
                "H0": "Two populations are equal",
                "H1": "Two populations are not equal"
            },

        "adf":
            {
                "H0": "There is a unit root in data (non-stationary)",
                "H1": "The time series is stationary (or trend-stationary)"
            },
        "kpss":
            {
                "H0": "The time series is stationary",
                "H1": "The time series is non-stationary"
            },
        "kendall":
            {
                "H0": "There is no association between the variables under study",
                "H1": "A trend (association  between the variables) exists"
            },
    }

    #* sample means
    #* 
    #* Mann-Whitney U Test.
    #* Wilcoxon Signed-Rank Test.
    #* Kruskal-Wallis H Test.
    #* Friedman Test.

    #* Relationship Between Variables
    #* Spearman's Rank Correlation.
    #* Kendall's Rank Correlation.
    #* Goodman and Kruskal's Rank Correlation.
    #* Somers' Rank Correlation.

    def __init__(
        self, 
        data: ArrLike,
        tests: str | list = "all"
    ) -> None:
        
        self._data = data
        self.tests = self._tests_adjustment(tests=tests)

        if self.tests is None:
            print("Wrong tests names were given. Default tests were set")
            self.tests = self._tests_adjustment(tests="all")
            

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    #! review required for method
    def _tests_adjustment(self, tests: str | list):
        #? can be moved to decorators
        if not isinstance(tests, str) and\
            not isinstance(tests, list):

            raise TypeError("Given formats cannot be processed")
        
        if isinstance(tests, str):
            return self.TESTS_NAMES
        
        return set(
            self.TESTS_NAMES.intersection(tests)
        )

    def hypothesis_check(
        self, 
        val, 
        alpha: None | float = None #* critical level
    ):
        if alpha is None:
            alpha = self.SIGNIFICANCE

        if val > alpha:
            return "Fail to reject H0 hypothesis", "H0"
        return "Reject H0 hypothesis -> H1 hypothesis", "H1"

    def run_tests(self, tests: list | None = None):

        if tests is None:
            tests = self.tests
        
        for i in tests:
            st, pvalue = getattr(self, i)()
            hypothesis_res, key = self.hypothesis_check(pvalue)
            hypothesis_desc  = self.HYPOTHESIS_DESC.get(i).get(key)

            self.print_res(
                pvalue=pvalue,
                test_name=i,
                hypothesis_res=hypothesis_res,
                hypothesis_desc=hypothesis_desc
            )
        

    def print_res(
        self,
        pvalue: float,
        test_name: str,
        hypothesis_res: str,
        hypothesis_desc: str
    ) -> None:
        print(
            f"Made <{test_name.upper()}> test:\n",
            f"{hypothesis_res}:\n",
            f"Verdict: {hypothesis_desc}\n"
            f"pvalue of test is {'{:.3f}'.format(pvalue)}\n",   
        )


    # def run_tests(self):
    #     return

    def correlation(self):
        return

class Traditional(HypothesisTests):
    #* QQ plot
    

    tests_names = [
        # "chisquare", #! parametric
        "normaltest",  #* D'Agostino's K2 Test
        "shapiro",  #* Shapiro-Wilk Test
        "anderson"  #* Anderson-Darling Test
        
    ]
    def __init__(
        self, 
        data: ArrLike,
        tests: str | list = "all"
    ):
        super().__init__(data, tests)


class Parametric(Traditional):
    #* QQ plot

    tests_names = [
        "chisquare", #* parametric
    ]

    def __init__(self):
        return
    
    def correlation(self):
        return
    
class NonParametric(Traditional):
    #* QQ plot

    TESTS_NAMES = [
        "normaltest",   #* D'Agostino's K2 Test
        "shapiro",      #* Shapiro-Wilk Test
        "anderson",     #* Anderson-Darling Test
        "kruskal",      #* Kruskal-Wallis H-test
        "wilcoxon",     #* Wilcoxon signed-rank test.
        "mannwhitneyu"  #* #* Mann-Whitney U rank test
    ]

    DISTRIBUTION_TESTS = [
        "normaltest",   #* D'Agostino's K2 Test
        "shapiro",      #* Shapiro-Wilk Test
        "anderson",     #* Anderson-Darling Test
    ]

    SAMPLE_TESTS = [
        "kruskal",      #* Kruskal-Wallis H-test
        "wilcoxon",     #* Wilcoxon signed-rank test.
        "mannwhitneyu"  #* #* Mann-Whitney U rank test
    ]

    TESTS_NAMES = [*SAMPLE_TESTS, *DISTRIBUTION_TESTS]

    def __init__(
        self, 
        data: ArrLike,
        tests: str | list = "all"
    ):
        super().__init__(data, tests)


    def _anderson_pvalue(self, st: float):

        st = st*(1 + (0.75/len(self.data)) + 2.25/(len(self.data)**2))

        if st >= 0.6:
            return np.exp(1.2939 - 5.709 * st + 0.0186 * st ** 2)
        
        elif 0.6 > st >= 0.34:
            return np.exp(0.9177 - 4.279 * st + 1.38 * st ** 2)

        elif 0.34 > st >= 0.2:
            return 1 - np.exp(-8.318 + 42.796 * st - 59.938 * st ** 2)

        else:
            return 1 - np.exp(-13.436 + 101.14 * st - 223.73 * st ** 2)
        

    def anderson(self, dist: str = "norm"):
        st, cvs, sls, *_ = stats.anderson(self.data, dist=dist)
        level = self.SIGNIFICANCE * 100  #* to % units

        sl_index =  list(sls).index(level)
        cv = cvs[sl_index]

        pvalue = self._anderson_pvalue(st)
        return st, pvalue

    def shapiro(self):
        return stats.shapiro(self.data)

    def normaltest(self):
        return stats.normaltest(self.data, nan_policy=self.NAN_POLICY)

    def kruskal(self):
        p1, p2 = self._split_data()
        return stats.kruskal(p1, p2, nan_policy=self.NAN_POLICY)

    def wilcoxon(self):
        st, pvalue = stats.wilcoxon(self.data, nan_policy=self.NAN_POLICY)
        return st, pvalue

    def mannwhitneyu(self):
        p1, p2 = self._split_data()
        return stats.mannwhitneyu(p1, p2, nan_policy=self.NAN_POLICY)

    def _split_data(self):
        half_len = int(len(self.data)/2)
        p1, p2 = self.data[: half_len], self.data[half_len :]
        return p1, p2

    def run_sample_tests(self) -> None:
        self.run_tests(tests=self.SAMPLE_TESTS)

    def run_distr_tests(self) -> None:
        self.run_tests(tests=self.DISTRIBUTION_TESTS)
        

class TimeSeries(HypothesisTests):
    #*  adf, kpss, Ljung-Box Test

    TESTS_NAMES = [
        "adf",      #* Augmented Dickey-Fuller unit root test
        "kpss"      #* Kwiatkowski-Phillips-Schmidt-Shin test
    ]

    def __init__(
        self, 
        data: ArrLike,
        tests: str | list = "all"
    ):
        super().__init__(data, tests)

    def adf(self, autolag="AIC"):
        st, pvalue, *_ = adfuller(self.data, autolag=autolag)
        return st, pvalue

    def kpss(self):
        st, pvalue, *_ = kpss(self.data)
        return st, pvalue

class NonStationaryTransforms(TimeSeries):
    #* Log, Power, BoxCox, Differencing transforms
    def __init__(self):
        return
