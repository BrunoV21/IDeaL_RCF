import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import norm
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import mixture
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns
from scipy.stats import kde
import copy


#Convenient functions for loading dataset
def loadCombinedArray(cases, field, dataset):
    data = np.concatenate([np.load('D:\\OneDrive - Universidade de Lisboa\\Turbulence Modelling Database\\'+dataset+'\\'+dataset+'_'+case+'_'+field + '.npy') for case in cases])
    return data

def loadCollumnStackFeatures(cases, fields, dataset):
    i = 0
    #print(fields)
    for field in fields:
        #print(field)
        data = loadCombinedArray_Turb_V(cases, field, dataset)
        if i == 0 :
            Features = data
        else : 
            Features = np.column_stack((Features, data))
        i += 1
    return Features

def loadLabels(cases, field):
    data = np.concatenate([np.load('D:\\OneDrive - Universidade de Lisboa\\Turbulence Modelling Database\\labels\\'+case+'_'+field + '.npy') for case in cases])
    return data

def loadCombinedArray_Turb_V(cases, field, dataset):
    data = np.concatenate([np.load(dataset+'_'+case+'_'+ field + '.npy') for case in cases])
    return data

def load_case(case, dataset, features, features_tensors, features_tensors_visc, labels_NL, labels, eV_labels,features_filter):
    x = loadCollumnStackFeatures([case], features, dataset)
    x = filter_features(x, features_filter)
    #x = transform_data(x, feat_list)
    basis = loadCombinedArray_Turb_V([case], features_tensors, dataset)
    Shat = loadCollumnStackFeatures([case], features_tensors_visc, dataset)
    #T_t = loadCollumnStackFeatures([case], ['T_t'], dataset)
    #Shat = Shat*T_t.reshape(T_t.shape[0],1)
    y_NL = loadCombinedArray_Turb_V([case], labels_NL, dataset)
    y = loadCombinedArray_Turb_V([case], labels, dataset)
    eV = loadCombinedArray_Turb_V([case], eV_labels, dataset)
    
    [Cx, Cy] = load_coordinates([case], 'komegasst')
    
    #y_cluster = loadCombinedArray_Turb_V([case], 'b', dataset)
    
    return [x, basis, Shat, y_NL, y, eV, Cx, Cy]

def Stack_Cases(case_dict, zone, index, cases):
    i = 0
      
    if zone == None:
        cases_g = [case for case in cases if len(case_dict[case][index]) > 0]
        #print(cases_g)
        for case in cases_g:
            if i == 0:
                stacked = case_dict[case][index]
            else:
                stacked = np.vstack((stacked, case_dict[case][index]))
            i += 1  
    else:
        for case in cases:
            if i == 0:
                stacked = case_dict[case][zone][index]
            else:
                stacked = np.vstack((stacked, case_dict[case][zone][index]))
            i += 1   

    return stacked

### same as stack cases but without zone
def Stack_Cases_pre(case_dict, cases, index):
    i = 0
    for case in cases:
        if i == 0:
            stacked = case_dict[case][index]
        else:
            stacked = np.vstack((stacked, case_dict[case][index]))
        i += 1   

    return stacked

def remove_outliers(Features):
    stdev = np.std(Features,axis=0)
    means = np.mean(Features,axis=0)
    ind_drop = np.empty(0)
    for i in range(len(Features[0,:])):
        ind_drop = np.concatenate((ind_drop,np.where((Features[:,i]>means[i]+5*stdev[i]) | (Features[:,i]<means[i]-5*stdev[i]) )[0]))
    return np.unique(ind_drop.astype(int))

def remove_outliers_data(x, basis, Shat, y):
    outlier_index = remove_outliers(x) #Features
    print('Found '+str(len(outlier_index))+' outliers in the input feature set')
    x = np.delete(x, outlier_index,axis=0)
    if bool(basis.any()):
        basis = np.delete(basis,outlier_index,axis=0)
    Shat = np.delete(Shat, outlier_index,axis=0) 
    y = np.delete(y,outlier_index,axis=0)
    return x, basis, Shat, y

def remove_outliers_test(x, basis, Shat, y, Cx, Cy):
    outlier_index = remove_outliers(x) #Features
    print('Found '+str(len(outlier_index))+' outliers in the input feature set')
    x = np.delete(x, outlier_index,axis=0)
    if bool(basis.any()):
        basis = np.delete(basis,outlier_index,axis=0)
    Shat = np.delete(Shat, outlier_index,axis=0)     
    y = np.delete(y,outlier_index,axis=0)
    Cx = np.delete(Cx,outlier_index,axis=0)
    Cy = np.delete(Cy,outlier_index,axis=0)
    return [x, basis, Shat, y, Cx, Cy, outlier_index]

def load_coordinates(case, dataset):
    Cx = loadCombinedArray(case, 'Cx', dataset)
    Cy = loadCombinedArray(case, 'Cy', dataset)
    return Cx, Cy

   
def build_scaler_labels(x, scaler):
    if not bool(scaler):
        #scaler = PowerTransformer()
        #scaler = RobustScaler()
        #scaler = MinMaxScaler()      
        #scaler = QuantileTransformer(output_distribution = 'normal')
        scaler = StandardScaler()
                
        print('Building Scaler on this Fold')
        x = scaler.fit_transform(x)
        
    else:
        
        print('Using previously built Scaler on this Fold')
        
        x = scaler.transform(x)
       
    return x, scaler

def build_scaler_eV(x, scaler):
    if not bool(scaler):
        scaler = PowerTransformer()
        #scaler = RobustScaler()
        #scaler = MinMaxScaler()      
        #scaler = QuantileTransformer(output_distribution = 'normal')
        #scaler = StandardScaler()
                
        print('Building Scaler on this Fold')
        x = scaler.fit_transform(x)
        
    else:
        
        print('Using previously built Scaler on this Fold')
        
        x = scaler.transform(x)
       
    return x, scaler

def build_scaler(x, scaler):
    if not bool(scaler):
        #scaler = PowerTransformer()
        #scaler = RobustScaler()
        scaler = MinMaxScaler()      
        #scaler = QuantileTransformer(output_distribution = 'normal')
        #scaler = StandardScaler()
                
        print('Building Scaler on this Fold')
        x = scaler.fit_transform(x)
        
    else:
        
        print('Using previously built Scaler on this Fold')
        
        x = scaler.transform(x)
       
    return x, scaler

def transform_data(x_train, feat_list, feats_not_to_transform = ['I1_2', 'I1_5', 'I1_8','I1_15','I1_17', 'I1_19']):
    for i in range(x_train.shape[1]) :
        if feat_list[i][0] != 'q':
            if feat_list[i] in feats_not_to_transform :
                print(f'>{feat_list[i]} not transformed as it is not skewed enough')
                continue            
            elif x_train[:, i].min() < 0 < x_train[:, i].max():
                #x_train[:, i] = np.cbrt(x_train[:, i])
                print(f'>{feat_list[i]} data positive and negative: skippnig') 
                continue
            else :     
                x_train[:, i] = np.log(abs(x_train[:, i])+1)
                print(f'>{feat_list[i]} data strictly positive or negative: applying log transformation')
    return x_train


def build_scalers(x_train, x_val, y_train, y_val, scaler, labels_scaler, features_filter):
    
    ### apply cubic root to x data only given that its negative and positive then scale
    x_train = transform_data(x_train, features_filter)
    x_val = transform_data(x_val, features_filter)
    
    [x_train, scaler] = build_scaler(x_train, scaler)
    [x_val, scaler] = build_scaler(x_val, scaler)
    
    [y_train, labels_scaler] = build_scaler(y_train, labels_scaler)
    [y_val, labels_scaler] = build_scaler(y_val, labels_scaler)
       
    
    return x_train, y_train, x_val, y_val, scaler, labels_scaler

def shuffle_data(x, basis, Shat, y_NL, y, eV):
    x, basis, Shat, y_NL, y, eV = shuffle(x, basis, Shat, y_NL, y, eV,  random_state = 42)
    return [x, basis, Shat, y_NL, y, eV]

def data_clean_case(case_dict, cases, feat_list, remove_outliers, shuffling, scaler, labels_scaler_NL, labels_scaler, labels_scaler_eV):
    local_case_dict = copy.deepcopy(case_dict)
    print('\n')
    if remove_outliers:
        print('==================== Removing Outliers ====================')
        local_case_dict = {case: remove_outliers_test(local_case_dict[case][0],
                                                      local_case_dict[case][1],
                                                      local_case_dict[case][2],
                                                      local_case_dict[case][3],
                                                      local_case_dict[case][4],
                                                      local_case_dict[case][5])[:6] for case in cases}
                           
    
    print('\n==================== Transforming Data ====================') 
    if len(cases) > 1:
        stacked_cases = [Stack_Cases_pre(local_case_dict, cases, i) for i in range(6)]
        stacked_cases[0] = transform_data(stacked_cases[0], feat_list)
        print(f'> Transformed data\n - x: {stacked_cases[0].shape}\n - basis: {stacked_cases[1].shape}\n - Shat: {stacked_cases[2].shape}\n - y: {stacked_cases[3].shape}\n')
        data_out = stacked_cases
    else:
        case = cases[0]
        local_case_dict[case][0] = transform_data(local_case_dict[case][0], feat_list)
        print(f'> Transformed data\n - x: {local_case_dict[case][0].shape}\n - basis: {local_case_dict[case][1].shape}\n - Shat: {local_case_dict[case][2].shape}\n - y: {local_case_dict[case][3].shape}\n')
        data_out = local_case_dict[case]
    
    print(data_out[0].shape)
    print('==================== Scaling Data ====================')

    try:
        data_out[0], scaler = build_scaler(data_out[0], scaler)
    except ValueError:
        print('> labels 0 have already been scaled')

    data_out[3], labels_scaler_NL = build_scaler_labels(data_out[3], labels_scaler_NL)
    data_out[4], labels_scaler = build_scaler_labels(data_out[4], labels_scaler)
    data_out[5], labels_scaler_eV = build_scaler_eV(data_out[5], labels_scaler_eV)
    # shuffle data
    if shuffling:
        data_out[:6] = shuffle_data(data_out[0], data_out[1], data_out[2], data_out[3], data_out[4], data_out[5])
        
    return data_out, scaler, labels_scaler_NL, labels_scaler, labels_scaler_eV

def build_index(features):
    indexes = []
    for feature in features:
        #print(feature)
        if feature[0] == 'I':
            if feature[1] == '1':
                indexes.append(int(feature[3:])-1)
            if feature[1] == '2':
                indexes.append(int(feature[3:])+19) 
        else:
            indexes.append(int(feature[-1])+39)
            
    return indexes  

def check_indexes(indexes, features, feat):
    i = 0
    for index in indexes:
        if features[i] == feat[index]:            
            print(f'element {i} is correct')
        else:
            print(features[index])
            print(feat[i])
            print(f'element {i} is not correct')
            break
        i += 1    

def filter_features(x, features_filter):
    if len(features_filter) == 0:
        x = x
    else:
        indexes_union = build_index(features_filter)
        x = x[:, indexes_union]
    return x    

   

def make_realizable(labels, indices):
    """ from J.Ling Github repo
    This function is specific to turbulence modeling.
    Given the anisotropy tensor, this function forces realizability
    by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
    Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
    constraints from first step, so this function should be called iteratively to get convergence
    to a realizable state.
    :param labels: the predicted anisotropy tensor (num_points X 9 array)
    """
    #numPoints = labels.shape[0]    
    
    A = np.zeros((3, 3))
    for i in indices:
        count = 0
        # Scales all on-diags to retain zero trace
        if np.min(labels[i, [0, 4, 8]]) < -1./3.:
            labels[i, [0, 4, 8]] *= -1./(3.*np.min(labels[i, [0, 4, 8]]))
            #print(1)
        elif 2.*np.abs(labels[i, 1]) > labels[i, 0] + labels[i, 4] + 2./3.:
            labels[i, 1] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
            labels[i, 3] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
            #print(2)
        elif 2.*np.abs(labels[i, 5]) > labels[i, 4] + labels[i, 8] + 2./3.:
            labels[i, 5] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
            labels[i, 7] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
            #print(3)

        elif 2.*np.abs(labels[i, 2]) > labels[i, 0] + labels[i, 8] + 2./3.:
            labels[i, 2] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
            labels[i, 6] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
            #print(4)
        else:
            count += 1
        
        # Enforce positive semidefinite by pushing evalues to non-negative
        A[0, 0] = labels[i, 0]
        A[1, 1] = labels[i, 4]
        A[2, 2] = labels[i, 8]
        A[0, 1] = labels[i, 1]
        A[1, 0] = labels[i, 1]
        A[1, 2] = labels[i, 5]
        A[2, 1] = labels[i, 5]
        A[0, 2] = labels[i, 2]
        A[2, 0] = labels[i, 2]
        evalues, evectors = np.linalg.eig(A)
        #print(evalues)
        #rint('\n')
        if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/2.:
            evalues = evalues*(3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/(2.*np.max(evalues))
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                labels[i, j] = A[j, j]
            labels[i, 1] = A[0, 1]
            labels[i, 5] = A[1, 2]
            labels[i, 2] = A[0, 2]
            labels[i, 3] = A[0, 1]
            labels[i, 7] = A[1, 2]
            labels[i, 6] = A[0, 2]
        elif np.max(evalues) > 1./3. - np.sort(evalues)[1]:
            evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                labels[i, j] = A[j, j]
            labels[i, 1] = A[0, 1]
            labels[i, 5] = A[1, 2]
            labels[i, 2] = A[0, 2]
            labels[i, 3] = A[0, 1]
            labels[i, 7] = A[1, 2]
            labels[i, 6] = A[0, 2]
        else:
            count += 1
                        
        if count == 2:
            #print(f'{i} already satisfies realizability')
            indices.remove(i)

    return labels, indices

def narrow_nonzero_trace(a, min_a):
    if min_a < -1/3:
        a = - a/(3*min_a)
    return a

def make_realizable_v2(labels, indices):
    #Schucman approach 
    # Algorythin implementation followed by Jiang et al.
    A = np.zeros((3, 3))
    for i in indices:
        
        count = 0
        
        A[0, 0] = labels[i, 0]
        A[1, 1] = labels[i, 4]
        A[2, 2] = labels[i, 8]
        A[0, 1] = labels[i, 1]
        A[1, 0] = labels[i, 1]
        A[1, 2] = labels[i, 5]
        A[2, 1] = labels[i, 5]
        A[0, 2] = labels[i, 2]
        A[2, 0] = labels[i, 2]
        
        #print(A)
        
        min_diag = min(A[0,0], A[1,1], A[2,2])
        if min_diag < -1/3:
            for i in range(3):
                A[i,i] = narrow_nonzero_trace(A[i,i], min_diag)
                #print(f'> nonzero traced narrowed')
            min_diag = min(A[0,0], A[1,1], A[2,2])
        else:
            #print('nonzero trace already is narrowed')
            count += 1
                  
        if (A[0,1]*A[0,1]) > (A[0,0]+1/3)*(A[1,1]+1/3):
            A[0,1] = np.sign(A[0,1])*np.sqrt(max((A[0,0]+1/3)*(A[1,1]+1/3), 0))
        else:
            count += 1
                  
       ### Compute eign values of A 1>2>3
                  
        eign_values, evectors = np.linalg.eig(A)
        [eign_3, eign_2, eign_1] =  np.sort(eign_values)
        
        if eign_1 < 0.5*(3*np.abs(eign_2)-eign_2):
            #amplify all eign_valuees
            #print(1)
            eign_values = eign_values*.5*(3*np.abs(eign_2)-eign_2)/eign_1
            [eign_3, eign_2, eign_1] =  np.sort(eign_values)
              
          
        elif eign_1 > 1/3-eign_2:
            #reduce all eign_values
            #print(2)
            eign_values = eign_values*(1/3-eign_2)/eign_1
            [eign_3, eign_2, eign_1] =  np.sort(eign_values)
            
        else:
            count += 1
            
        if count == 3:
            #print(f'{i} already satisfies realizability')
            indices.remove(i)
              
        # build new A from new eign and old evectors

        A_ = np.dot(np.dot(evectors, np.diag(eign_values)), np.linalg.inv(evectors))

        A[0,1] = 0.5*(A_[0,1] + A_[1,0])
        A[1,0] = 0.5*(A_[1,0] + A_[0,1])
        
        #print(A)
        
        labels[i, 0] = A[0,0]
        labels[i, 1] = A[0,1]
        labels[i, 2] = A[0,2]
        labels[i, 3] = A[1,0]
        labels[i, 4] = A[1,1]
        labels[i, 5] = A[1,2]
        labels[i, 6] = A[2,0]
        labels[i, 7] = A[2,1]
        labels[i, 8] = A[2,2]
        
    return labels, indices        
        
        
def force_bounded(b, max_dict, min_dict):
    count_max = 0
    count_min = 0
    
    for i in range(len(b)):
        for j in range(4):
            if b[i,j] >= max_dict[j]:
                b[i,j] = max_dict[j]
                if i == 9702:
                    print('\n\ngotcha\n\n')
                count_max += 1
            elif b[i,j] <= min_dict[j]:
                b[i,j] = min_dict[j]
                count_min += 1
                
    print(f'\n{count_max} points were bounded to the upper limit\n{count_min} points were bounded to the lower limit\n')
    
    return b

def force_realizability(b):
    zeros = np.zeros(len(b))
    #print(np.argmin(b[:,0]))
    
    max_dict = {}
    min_dict = {}
    
    for i in range(4):
        max_dict[i] = np.max(b[:,i])
        min_dict[i] = np.min(b[:,i])
    
    
    b = np.column_stack((
       b[:,0], b[:,1], zeros,
       b[:,1], b[:,2], zeros,
       zeros,  zeros,  b[:,3]))
    
    
    indices = [i for i in range(b.shape[0])]
             
    i = 1
    previous_len = -5
  
    #print(b)
    #b, indices = make_realizable_v2(b, indices)
    
    #print(b)
    
    while True:
        b, indices = make_realizable_v2(b, indices)
        print(f'iteration {i}\n> {b.shape[0] - len(indices)} out of {b.shape[0]} points already satisfy realizability\n')
        if previous_len == b.shape[0] - len(indices) :
            break
        else:
            previous_len = b.shape[0] - len(indices) 
        i += 1
        #if i >= 2:
         #   break
        
    b = np.delete(b.reshape((len(b),9)),[2, 3 , 5, 6, 7],axis=1)
    
    #print(np.min(b[:,0]))
    #print(np.argmin(b[:,0]))
    
    #b = force_bounded(b, max_dict, min_dict)
    
    #print(np.min(b[:,0]))
    #print(np.argmin(b[:,0]))
    
    print(' ================================================ \n')
    
    return b

