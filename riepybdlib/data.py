from pkg_resources import resource_listdir
from pkg_resources import resource_filename
import random
import string

import scipy.io as sio
import numpy as np

def get_letter_data(letter='', n_samples = -1, n_derivs=0,use_time=True, tmax=10):
    '''Function returns a list of demonstrations of a random 2D-letter. 
    
    The data of each demonstration is a numpy array with a horizontal concatenation of 
    time, position, and velocity:
    [t,pos,vel]

    Optional arguments:
    letter   :  The desired letter                  (default: random)
    n_samples: The desired number of samples        (default: maximum available)
    n_derivs : The number of time derivatives to include (default: 1 (velocity included)
    use_time : Flag to indicate if time signal should be included (default: True)
    tmax     : The maximum duration of the movement (default: 10)


    '''

    if letter == '':
        letter=random.choice(string.ascii_letters)



    fn = resource_filename(__name__,'data/2Dletters/%s.mat'%(letter.upper()))
    tmax = 10

    # Load the mat files:
    data_contents =sio.loadmat(fn)

    # Get demos struct:
    mat_struct = data_contents['demos']

    if n_samples == -1:
        n_samples = mat_struct.shape[1]

    data = []

    for i in range(0,n_samples):       
        val = mat_struct[0,i]
        tmp = []
        pos = val['pos'][0,0]

        # Time signal
        if use_time == True:
            t = np.linspace(0,tmax,pos.shape[1]).reshape((1,pos.shape[1]))
            tmp = np.vstack((t,pos))
        else:
            tmp = pos

        # Spatial information:
        if n_derivs >2 or n_derivs<0:
            raise ValueError('Specified n_derivs (%i) not available'%(n_derivs))
        if n_derivs>=1:
            vel = val['vel'][0,0]
            tmp = np.vstack((tmp,vel))
        if n_derivs==2:
            acc = val['acc'][0,0]
            tmp = np.vstack((tmp,acc))

        # Transpose data to comply with standord of having 
        # rows of data points
        data.append(tmp.T)
    return data

def get_tp_frommat(praw,use_time=False):
    plib = []
    for i in range(praw['A'].shape[1]):
        p = {}
        p['A'] = praw['A'][0,i]
        
        # Check determinant:
        for v in range(p['A'].shape[1]):
            p['A'][:,v] =p['A'][:,v]/np.linalg.norm(p['A'][:,v])
        if np.linalg.det(p['A'])!=1:
            #print('flip')
            p['A'][:,0] = -p['A'][:,0]

        p['b'] = praw['b'][0,i][:,0]

        # Add time transformation:
        if use_time:
            # Add time index to task-parameters
            Atmp = np.eye(p['A'].shape[0]+1)
            Atmp[1:,1:] = p['A']
            p['A'] = Atmp
            p['b'] = np.hstack((np.array([0]),p['b']))
        
        # For Debugging:
        #print('A:\n',p['A'])
        #print('det(A):',np.linalg.det(p['A']))
        #print('b: ', p['b'])
        plib.append(p)
    return plib



def get_tp_data(use_time=False):

    # Load task-parameterized data:
    fn = resource_filename(__name__,'data/Data01.mat')
    data_contents = sio.loadmat(fn)

    # The data in the different frames:
    data = data_contents['Data']
    (n_vars,n_frames,n_data) = data.shape
    if use_time:
        n_vars+=1

    # The frame information:
    s = data_contents['s']
    (_,n_samples) = s.shape
    n_data = int(n_data/n_samples)


    TPs = []
    DataNew = np.zeros((n_samples,n_frames,n_data,n_vars))

    Data0= []
    for n in range(n_samples):
        stmp = s[0,n]
        # Data in global frame:
        Data0n = stmp[1]
        Data0.append(Data0n)
        
        # TPs:
        TPs.append(get_tp_frommat(stmp[0],use_time) )

        for m,TP in enumerate(TPs[n]):
            # Reconstruct local data using task-parameters:
            if use_time:
                DataNew[n,m,:]  =  np.linalg.solve(TP['A'],(Data0n.T- TP['b']).T).T
            else:
                DataNew[n,m,:]  =  np.linalg.solve(TP['A'],(Data0n[1:,:].T- TP['b']).T).T

        
    # Concatenate samples in new format
    data = DataNew[0,:]
    data0 = Data0[0]
    for i in range(n_samples-1):
        data = np.concatenate((data,DataNew[i+1,:]),axis=1)
        data0 = np.hstack((data0,Data0[i+1]))
    data0 = data0.T

    data_info ={}
    
    data_info['n_data'] = n_data
    data_info['n_samples'] = n_samples
    data_info['n_frames'] = n_frames
    data_info['n_vars'] = n_vars 
    
    return (data,data0,TPs,data_info)



