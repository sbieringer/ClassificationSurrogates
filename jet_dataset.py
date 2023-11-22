'''
The dataset wrapper.

'''



from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class JetDataset(Dataset):
    def __init__(self, path=None, mode=None, saved_path=None, only_pink=False, del_context=[]):
        '''
        setup to return:
        - the conditioning features, normalised <one vector>
        - the target output (jet_p_top_ParT_full)
        '''
        super(Dataset, self).__init__()

        #hard coded normalisation to enforce consistency
        means = np.array([[ 5.00000000e-01,  6.23166764e+02,  7.92356175e-05, -3.36014642e-04,
        9.06941547e+02,  4.30536727e+01,  1.07352269e+02,  1.84643913e-01,
        8.86405459e-02,  4.94997103e-02,  3.85121050e-02,  1.37753736e-05,
       -1.05942269e-04,  3.21420000e-02,  3.18712449e+02,  3.07605615e+01]], dtype='float32')
        
        std = np.array([[5.00000000e-01, 1.09143856e+02, 8.75131335e-01, 1.81404007e+00,
       3.49699836e+02, 1.69453641e+01, 7.60797433e+01, 1.18626813e-01,
       5.76037338e-02, 2.98865181e-02, 2.16299983e-02, 6.10334641e-01,
       1.28264871e+00, 4.24251893e+00, 3.29859845e+02, 1.92426810e+01]], dtype='float32')

        assert mode == 'train' or mode == 'test' or mode == 'val' or mode is None    
        if mode == 'train':
            assert path is not None
            self.data = pd.read_hdf(path+"/filtered_jetclass_train.h5", key="df")
        elif mode == 'val':
            assert path is not None
            self.data = pd.read_hdf(path+"/filtered_jetclass_val.h5", key="df")
        elif mode == 'test':
            assert path is not None
            self.data = pd.read_hdf(path+"/filtered_jetclass_test.h5", key="df")
        else:
            assert saved_path is not None
            self.load(saved_path)
            return

        #to be adapted
        def use_data(k):
            use = not (k=='jet_p_top_ParT_full' or k=='jet_p_top_ParT_kin' or k in del_context) 
            return use

        #preprocess
        self.features = np.concatenate(
            [np.array(self.data[k],dtype='float32')[...,np.newaxis] for k in self.data.keys() 
             if use_data(k)],axis=-1
        )

        keys = [k for k in  self.data.keys() if not (k=='jet_p_top_ParT_full' or k=='jet_p_top_ParT_kin')]
        means = np.array([[means[0,i] for i,k in enumerate(keys) if use_data(k)]])
        std = np.array([[std[0,i] for i,k in enumerate(keys) if use_data(k)]])
        
        self.means_norm = means
        self.std_norm = std
        self.keys = keys

        self.features = (self.features - means) / std
        self.truth = np.array(self.data['aux_genpart_pid'],dtype='float32')[...,np.newaxis]
        self.truth = np.where(np.abs(self.truth) == 6, 1, np.zeros_like(self.truth))
        self.target = np.array(self.data['jet_p_top_ParT_full'],dtype='float32')[...,np.newaxis]

        self.raw_target = self.target 
        self.target = np.log(self.target/(1.-self.target + 1e-9))/20. #invert sigmoid

        self.raw_target_gen = None
        self.target_gen = None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return  self.target[index], self.features[index]
    
    def load(self, filename):
        import pickle
        import gzip
        with gzip.open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)       
        self.__dict__.update(tmp_dict) 

    def save(self, filename):
        import pickle
        import gzip
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

#from sklearn.metrics import roc_curve
#
#jd = JetDataset("./jet_data",'val')
