from typing import Optional
import numpy as np


class MakeRealizable(object):
    """
    inspired from [PiResNet-2.0: Data-driven Turbulence Modeling](https://github.com/Jackachao0618/PiResNet-2.0)
    """

    def __init__(self, debug :Optional[bool]=False) -> None:
        self.debug = debug


    def force_realizability(self, a :np.array):
        zeros = np.zeros(len(a))        
        max_dict = {}
        min_dict = {}        
        for i in range(a.shape[1]):
            max_dict[i] = np.max(a[:,i])
            min_dict[i] = np.min(a[:,i])        
        
        a = np.column_stack(
            (
                a[:,0], a[:,1], zeros,
                a[:,1], a[:,2], zeros,
                zeros,  zeros,  a[:,3]
            )
        )
        indices = [i for i in range(a.shape[0])]
        i = 1
        previous_len = -5
        while True:
            a, indices = self.make_realizable(a, indices)
            if self.debug:
                print(f'iteration {i}\n> {a.shape[0] - len(indices)} out of {a.shape[0]} points already satisfy realizability\n')
            if previous_len == a.shape[0] - len(indices) :
                break
            else:
                previous_len = a.shape[0] - len(indices) 
            i += 1            
        a = np.delete(a.reshape((len(a),9)),[2, 3 , 5, 6, 7],axis=1)        
        print(' ================================================ \n') if self.debug else ...        
        return a


    def narrow_nonzero_trace(self, a, min_a):
        if min_a < -1/3:
            a = - a/(3*min_a)
        return a


    def make_realizable(self, labels, indices):
        ### Schucman approach 
        ### Algorythin implementation followed by Jiang et al.
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
            min_diag = min(A[0,0], A[1,1], A[2,2])
            if min_diag < -1/3:
                for i in range(3):
                    A[i,i] = self.narrow_nonzero_trace(A[i,i], min_diag)
                min_diag = min(A[0,0], A[1,1], A[2,2])
            else:
                count += 1
            if (A[0,1]*A[0,1]) > (A[0,0]+1/3)*(A[1,1]+1/3):
                A[0,1] = np.sign(A[0,1])*np.sqrt(max((A[0,0]+1/3)*(A[1,1]+1/3), 0))
            else:
                count += 1                    
            ### Compute eign values of A 1>2>3                    
            eign_values, evectors = np.linalg.eig(A)
            [eign_3, eign_2, eign_1] =  np.sort(eign_values)            
            if eign_1 < 0.5*(3*np.abs(eign_2)-eign_2):
                ### amplify all eign_valuees
                eign_values = eign_values*.5*(3*np.abs(eign_2)-eign_2)/eign_1
                [eign_3, eign_2, eign_1] =  np.sort(eign_values)            
            elif eign_1 > 1/3-eign_2:
                ### reduce all eign_values
                eign_values = eign_values*(1/3-eign_2)/eign_1
                [eign_3, eign_2, eign_1] =  np.sort(eign_values)                
            else:
                count += 1                
            if count == 3:
                indices.remove(i)
                
            ### build new A from new eign and old evectors
            A_ = np.dot(np.dot(evectors, np.diag(eign_values)), np.linalg.inv(evectors))

            A[0,1] = 0.5*(A_[0,1] + A_[1,0])
            A[1,0] = 0.5*(A_[1,0] + A_[0,1])
            
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