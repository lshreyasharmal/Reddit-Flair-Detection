import numpy as np
def get_given_features_from_data(feature_list, data):
    final_data = np.array([x for x in data[feature_list[0]]])
    final_data = np.atleast_2d(final_data)
    for i in range(1,len(feature_list)):
        feature = feature_list[i]
        data_features = np.array([x for x in data[feature]])
        reshaped_array = np.atleast_2d(data_features)
        if(reshaped_array.shape[0]==1 and reshaped_array.shape[1]==1):
            reshaped_array = reshaped_array.T
        final_data = np.concatenate((final_data,reshaped_array),axis=1)
    return final_data
