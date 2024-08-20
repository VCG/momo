import pandas as pd
import numpy as np

class FeatureVector():
    def __init__(self):
        self.descriptors = []
        self.values = []

    def append_value(self, descriptor, value):
        assert descriptor not in self.descriptors
        self.descriptors.append(descriptor)
        self.values.append(value)
    
    def get_value(self, descriptor):
        assert descriptor in self.descriptors
        return self.values[self.descriptors.index(descriptor)]

    def is_empty(self):
        return len(self.descriptors) == 0
    
    def to_pandas(self):
        return pd.DataFrame({"feature_descriptor" : self.descriptors, "value" : self.values})


class GroupFeatures():
    def __init__(self):
        self.group_member_ids = []
        self.group_member_features = []
        self.stats = pd.DataFrame(columns=['feature_descriptor','mean','std', "min", "max", "Q25", "median", "Q75"]) 

    def add_group_member(self, member_features, member_id=None, calculate=False):
        assert isinstance(member_features, FeatureVector)
        self.group_member_ids.append(member_id)
        self.group_member_features.append(member_features)

        if(calculate):
            self.calculate()
        
    def get_common_features(self):
        if(not len(self.group_member_features)):
            return []
        
        descriptors = set(self.group_member_features[0].descriptors)
        for k in range(1, len(self.group_member_features)):
            descriptors &= set(self.group_member_features[k].descriptors)

        return list(sorted(descriptors))

    def calculate(self):
        df_cols = self.stats.columns
        self.stats = pd.DataFrame(columns=df_cols)

        common_features = self.get_common_features()
        for feature_descriptor in common_features:
            row = [feature_descriptor]

            values = [member_features.get_value(feature_descriptor) for member_features in self.group_member_features]
            row.append(np.mean(values))
            row.append(np.std(values))
            row.append(np.min(values))
            row.append(np.max(values))
            row.append(np.quantile(values, 0.25))
            row.append(np.median(values))
            row.append(np.quantile(values, 0.75))

            if(self.stats.empty):
                self.stats = pd.DataFrame([row], columns=df_cols)
            else:
                self.stats = pd.concat([self.stats, pd.DataFrame([row], columns=df_cols)], ignore_index=True)    


class DeltaFeatures:
    def __init__(self, group_1, group_2):
        self.stat_columns = group_1.stats.columns[1:].to_list()
        assert self.stat_columns == group_1.stats.columns[1:].to_list()

        self.group_1 = group_1
        self.group_2 = group_2
        self.calculate()
        
    def calculate(self):
        common_features = set(self.group_1.stats.feature_descriptor.values) & set(self.group_2.stats.feature_descriptor.values)
        
        stats_1 = self.group_1.stats[self.group_1.stats.feature_descriptor.isin(common_features)]
        stats_2 = self.group_2.stats[self.group_2.stats.feature_descriptor.isin(common_features)]
        assert stats_1.feature_descriptor.to_list() == stats_2.feature_descriptor.to_list()

        self.signed_deviations = stats_1.copy()
        self.signed_deviations.iloc[:,1:] = stats_2.iloc[:,1:] - stats_1.iloc[:,1:]

        self.signed_relative_deviations = stats_1.copy()
        self.signed_relative_deviations.iloc[:,1:] = np.divide(stats_2.iloc[:,1:] - stats_1.iloc[:,1:], stats_1.iloc[:,1:])