import warnings
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *


class Astronaut():

    def __init__(self, data, indicator_mapping):
        self.data = data
        self.indicator_mapping = indicator_mapping

    @classmethod
    def from_excel(cls, xlsx_filepath, indicator_evaluations_filepath):
        """Instantiate the class from an excel file.
        
        Args:
            xlsx_filepath: Path to the excel file as exported from Kosmos.
            indicator_evaluations_filepath: Path to the JSON file containing the evalutations of the indicator variables. 
        """
        df = pd.read_excel(xlsx_filepath)
        df.iloc[0] = df.iloc[0].ffill()
        df = df.iloc[:3, 1:]

        detailgoal = pd.Series(df.columns)
        detailgoal[detailgoal.str.startswith('Unnamed')] = None
        detailgoal = detailgoal.ffill()

        data = pd.DataFrame({
            'Detailziel': detailgoal,
            'Kennzahl': df.iloc[0].reset_index(drop=True),
            'Jahr': df.iloc[1].reset_index(drop=True),
            'Wert': df.iloc[2].reset_index(drop=True),
        })

        data['Jahr'] = pd.to_datetime(data['Jahr'].astype(str).str[:4])

        try:
            data['Wert'] = pd.to_numeric(data['Wert'].astype(str))
        except ValueError:
            data['Wert'] = data['Wert'].astype(str).str.replace(',', '')
            data['Wert'] = pd.to_numeric(data['Wert'])

        # Turn each indicator into it's own column.
        data = data.pivot(index='Jahr', columns='Kennzahl', values='Wert')

        indicator_evaluations = json.load(open(indicator_evaluations_filepath))
        
        return cls(data, indicator_evaluations)

    def plot_indicators(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for indicator in self.data.columns:
                self.plot_indicator(indicator)

    def plot_indicator(self, indicator):
        (ggplot(self.data, aes(x='self.data.index', y=indicator))
         + geom_point(color='red', na_rm=True)
         + geom_line(color='red')
         + geom_smooth(method='lm')
         + xlab('Jahr')
         + scale_x_date(date_breaks='1 year', date_labels='%Y')
         ).draw()

    def plot_correlation_matrix(self):
        data_series_only = self._remove_single_value_indicators()
        
        correlation_matrix = data_series_only.corr()
        # Generate a mask for the upper triangle
        mask = np.zeros_like(correlation_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        ticks = [-1, -0.5, 0, 0.5, 1]
        # Draw the heatmap with the mask and correct aspect ratio
        heatmap = sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=0.5, cbar_kws={'shrink': 0.5, 'ticks': ticks})
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=35, ha='right')
        
    def _remove_single_value_indicators(self):
        data_series_only = self.data.copy()

        for indicator in data_series_only.columns:
           if data_series_only[indicator].notna().sum() <= 1:
               del data_series_only[indicator]
        
        return data_series_only

    def compute_score(self):
        pass

    def plot_trend(self):
        data_adjusted = self._adjust_data()
        trend = self._diff_and_threshold(data_adjusted)
        trend_avg = trend.mean(axis='columns')
        n_values_per_year = (trend.notna()).sum(axis='columns')

        weighted_trend_avg = pd.DataFrame({
            'year': trend_avg.index,
            'trend': trend_avg,
            'n datapoints': n_values_per_year

        }).dropna()

        self._create_trend_plot(weighted_trend_avg)

    def _create_trend_plot(self, weighted_trend_avg):
        (ggplot(weighted_trend_avg, aes(x='year', y='trend', size='n datapoints'))
         + geom_point()
         + geom_line(size=1)
         + geom_hline(aes(yintercept=0), linetype='--')
         + scale_x_date(date_breaks='1 year', date_labels='%Y')
         + ylim(-1, 1)
         ).draw()

    def _diff_and_threshold(self, data_adjusted):
        diff = data_adjusted.diff()
        trend = diff > 0
        trend[diff.isna()] = np.nan
        trend[trend == 0] = -1
        return trend

    def _adjust_data(self):
        data_adjusted = self.data.copy()
        for indicator, desired_direction in self.indicator_mapping.items():
            if desired_direction == 'negative':
                data_adjusted[indicator] *= -1
            elif desired_direction == 'neutral':
                del data_adjusted[indicator]
        return data_adjusted
