'''
Martijn Zeestraten, August 2016

'''

import numpy as np
from copy import deepcopy
from copy import copy

import riepybdlib.angular_representations as ar


# ---------------- Manifold mapping functions
# Euclean space Mappings:
def eucl_action(x,g,h):
    return x - g + h

def eucl_exp_e(g_tan,reg=None):
    return g_tan

def eucl_log_e(g,reg=None):
    return g

def eucl_parallel_transport(Xg, g, h, t=1):
    return Xg


# -----------------     Quaternion Action, Log and Exponential        
def arccos_star(rho):
    if type(rho) is not np.ndarray:
        # Check rho
        if abs(rho)>1:
            # Check error:
            if (abs(rho) - 1 > 1e-6):
                print('arcos_star: abs(rho) > 1+1e-6:'.format( abs(rho)-1) )
                
            # Fix error
            rho = 1*np.sign(rho)
        
        # Single mode:
        if (-1.0 <= rho and rho < 0.0):
            return np.arccos(rho) - np.pi
        else:
            return np.arccos(rho)
    else:
        # Batch mode:
        rho = np.array([rho])
        
        ones = np.ones(rho.shape)
        rho = np.max(np.vstack( (rho,-1*ones ) ), axis=0)
        rho = np.min(np.vstack( (rho, 1*ones ) ), axis=0)

        acos_rho = np.zeros(rho.shape)
        sl1 = np.ix_ ((-1.0 <= rho)*(rho < 0.0)==1)
        sl2 = np.ix_ ((1.0 > rho)*(rho >= 0.0)==1)

        acos_rho[sl1] = np.arccos(rho[sl1]) - np.pi
        acos_rho[sl2] = np.arccos(rho[sl2])

        return acos_rho

quat_id = ar.Quaternion(1, np.zeros(3) )

def quat_action(x,g,h):
    return h*g.i()*x

def quat_log_e(g, reg=1e-6):
    d_added = False
    if type(g) is list:
        # Batch mode:
        g_np = ar.Quaternion.to_nparray_st(g)
    
        # Create tangent values, and initalize to zero
        g_tan = np.zeros( (g_np.shape[0], 3) )

        # compute tangent values for quaternions which have q0 other than 1.0 - reg
        # Slices:
        g_np0_abs = abs(g_np[:,0]-1.0) > reg
        sl_123 = np.ix_(g_np0_abs, range(1,4) )
        sl_012 = np.ix_(g_np0_abs, range(3) )
        sl_0   = np.ix_(g_np0_abs, [0] )

        # Compute tangent values:
        acos_q0 = arccos_star(g_np[sl_0][:,0])
        qnorm   = g_np[sl_123].T/ np.linalg.norm(g_np[sl_123], axis=1)
        g_tan[sl_012] = (qnorm*acos_q0).T

        return g_tan
    else:
        # Single mode:
        if abs(g.q0 - 1.0)>reg:
            return arccos_star(g.q0)* (g.q/np.linalg.norm(g.q))
        else:
            return np.zeros(3)
    
    
# The exponential map:
def quat_exp_e(g_tan, reg=1e-6):
    if g_tan.ndim == 2:
        # Batch mode:
        qnorm = np.linalg.norm(g_tan, axis=1)
        qvec = np.vstack( (np.ones( g_tan.shape[0]), np.zeros( g_tan.shape ).T) ).T

        # Select the non identity quaternions:
        qnorm_abs = abs(qnorm) > reg
        sl_0123 = np.ix_( qnorm_abs, range(4) )
        sl_012 =  np.ix_( qnorm_abs, range(3) )
        sl     =  np.ix_( qnorm_abs)
        qnorm = qnorm[sl] 

        # Compute the non identity values:
        qvec[sl_0123]  = np.vstack( (np.cos(qnorm), g_tan[sl_012].T*(np.sin(qnorm)/qnorm) ) ).T

        # Generate resulting quaternions:
        res = []
        for i in range(g_tan.shape[0]):
            res.append(ar.Quaternion(qvec[i,0], qvec[i,1:]) )  # Return result:
        #if d_added:
        #    return res[0] # return on the single quaternion that was requested
        #else:
        return res # Return the list of quaternions
    else:
        # Single mode:
        qnorm = np.linalg.norm(g_tan)
        if ( qnorm != 0 ):
            return ar.Quaternion( np.cos(qnorm), np.sin(qnorm)*( g_tan/qnorm )) 
        else:
            return ar.Quaternion(1, np.zeros(3) )

def quat_log(x,g, reg=1e-6):
    return quat_log_e(g.i()*x, reg)

def quat_exp(x,g, reg=1e-6):
    return g*quat_exp_e(x, reg)

def quat_parallel_transport(Xg, g, h, t=1):
    ''' Parallel transport of vectors in X from h to g*t, 0 <= t <= 1
        Implementation is modified version of the one reported in
        Optimization algorithms on Manifolds:
    '''
    
    # Get intermediate position on geodesic between g and h 
    if t<1:
        ht = quat_exp( quat_log(h, g)*t, g)
    else:
        ht=h
    
    # Get quaternion matrices for rotation computation
    Qeg = g.Q()  # Rotation between origin and g
    Qeh = ht.Q() # between  rotation between ht and origin
    
    
    # Get tangent  vector of h in g, expressed in R^4
    # We first construct it at the origin (by augmenting it with 0)
    # and then rotate it to point g
    v = Qeg.dot(np.hstack([[0], quat_log(h,g)]))
        
    # Transform g into np array for computations
    gnp = g.to_nparray()  # Transform to np array
    m = np.linalg.norm(v) # Angle of rotation (or 'transport)
    
    # Compute Tangential rotation (this is done in n_dimM)
    if m < 1e-6:
        Rgh = np.eye(4)
    else:
        u = (v/m)[:,None]
        gnp = gnp[:,None]

        Rgh = (- gnp*np.sin( m*t )*u.T 
             + u*np.cos( m*t )*u.T
             + ( np.eye(4) - u.dot(u.T) )
            )
        
    # ----- Finally compute rotation compensation to achieve parallel transport:
    Ie = np.eye(4)[:,1:].T
    
    Ig   = Ie.dot(Qeg.T)                   # Base orientation at g
    Ih   = Ig.dot(Rgh.T)                   # Base orientation at h by parallel transport
    Ie_p = Ih.dot(Qeh)                     # Tangent space orientation at origin with parallel transport
    
    # Compute relative rotation:
    R = Ie.dot(Ie_p.T)
    
    if np.sign(np.trace(R)) ==-1:
        # Change sign, to ensure proper rotation
        R = -R
                   
    # Transform points and return
    return Xg.dot(R.T)    
        
# ----------------------  S^2,
s2_id = np.array([0,0,1])
def s2_action(x, g, h):
    ''' Moves x relative to g, to y relative to h

    '''
    # Convert possible list into nparray
    if type(x) is list:
        x = np.vstack(x)

    # Get rotation of origin e to g
    ax, angle = ar.get_axisangle(g)
    Reg = ar.R_from_axis_angle(ax, angle)

    # Get rotation of origin e to h
    ax, angle = ar.get_axisangle(h)
    Reh = ar.R_from_axis_angle(ax, angle)

    # Creat rotation that moves x from g to the origin e,
    # and then from e to h:
    A = Reh.dot(Reg.T)

    return x.dot(A.T)

def s2_exp_e(g_tan, reg=1e-6):
    if g_tan.ndim ==2:
        # Batch operation:
    
        # Standard we assume unit values:
        val = np.vstack( (
                          np.zeros( (2, g_tan.shape[0]) ), 
                          np.ones( g_tan.shape[0] ) 
                         ) 
                       ).T
        
        # Compute distance for all values that are larger than the regularization:
        norm = np.linalg.norm(g_tan,axis=1)
        cond = norm> reg
        sl_2 =  np.ix_( cond, [2] )
        sl_01 = np.ix_( cond, range(0,2) )

        norm = norm[np.ix_(cond)]         # Throw away the norms we don't use
        gt_norm = (g_tan[sl_01].T/norm).T # Normalize the input vector of selected values

        val[sl_2]  = np.cos(norm)[:,None]
        val[sl_01] = (gt_norm.T*np.sin(norm)).T

        return val 
    else:
        # Single mode:
        norm = np.linalg.norm(g_tan)
        if norm > reg:
            gt_norm = g_tan/norm
            return np.hstack( (np.sin(norm)*gt_norm, [np.cos(norm)]) )
        else:
            return np.array([0,0,1])
    
def s2_log_e(g, reg = 1e-10):
    '''Map values g, that lie on the Manifold, into the tangent space at the origin'''
    
    # Check input
    d_added = False
    if g.ndim ==2:
        # Batch operation,
        # Assume all values lie at the origin:
        val = np.zeros( (g.shape[0], 2) )

        # Compute distance for all values that are larger than the regularization:
        cond =  (1-g[:,2]) > reg
        sl_2 =  np.ix_( cond, [2] )
        sl_01 = np.ix_( cond, range(0,2) )

        val[sl_01] = (g[sl_01].T*( np.arccos( g[sl_2])[:,0]/np.linalg.norm(g[sl_01], axis=1) ) ).T
        return val 
    else:
        # single mode:
        if abs(1-g[2]) > reg:
            return np.arccos(g[2])*g[0:2]/np.linalg.norm(g[0:2])
        else:
            return np.array([0,0])

def s2_exp(x, g, reg=1e-10):

    if type(x) is list:
        x = np.vstack(x)

    # Get rotation of origin e to g
    ax, angle = ar.get_axisangle(g)
    Reg = ar.R_from_axis_angle(ax, angle)

    return s2_exp_e(x,reg).dot(Reg.T)

def s2_log(x, g, reg=1e-10):

    if type(x) is list:
        x = np.vstack(x)

    # Get rotation of origin e to g
    ax, angle = ar.get_axisangle(g)
    Reg = ar.R_from_axis_angle(ax, angle)

    return s2_log_e(x.dot(Reg), reg)

def s2_parallel_transport(Xg, g, h, t=1):
    ''' Parallel transport of vectors in X from h to g*t, 0 <= t <= 1
        Implementation is modified version of the one reported in
        Optimization algorithms on Manifolds:
        Xg : array of tangent vectors to parallel tranport  n_data x n_dimT
        g  : base of tangent vectors                        n_dimM
        h  : final point of tangent vectors                 n_dimM
        t  : position on curve between g and h              0 <= t <= 1
    '''
    
    
    # Compute final point based on t
    if t<1:
        ht = s2_exp( s2_log(h, g)*t, g) # current position of h*t in on manifold
    else:
        ht=h
    
    # ----- Compute rotations between different locations:
    
    # Rotation between origin and g
    (ax, angle) = ar.get_axisangle(g)
    Reg = ar.R_from_axis_angle(ax,angle)
    
    # Rotation between origin and ht (in the original atlas)
    (ax, angle) = ar.get_axisangle(ht)
    Reh  = ar.R_from_axis_angle(ax,angle)  # Rotation between final point and origin
    
    # ------ Compute orientation at ht using parallel transpot
    v  = Reg.dot( np.hstack([s2_log(h,g),[0]]) ) # Direction vector in R3
    m  = np.linalg.norm(v)  # Angle of rotation
    
    # Compute Tangential rotation (this is done in n_dimM)
    if m < 1e-10:
        Rgh = np.eye(3)
    else:
        u = (v/m)[:,None]
        g = g[:,None]

        Rgh = (- g*np.sin( m*t )*u.T 
             + u*np.cos( m*t )*u.T
             + ( np.eye(3) - u.dot(u.T) )
            )
        
    # ----- Finally compute rotation compensation to achieve parallel transport:
    Ie = np.eye(3)[:,0:2].T
    Ig   = Ie.dot(Reg.T)                   # Base orientation at g
    Ih   = Ig.dot(Rgh.T)                   # Base orientation at h by parallel transport
    Ie_p = Ih.dot(Reh)                     # Tangent space orientation at origin with parallel transport
    
    # Compute relative rotation:
    R = Ie.dot(Ie_p.T)
    
                   
    # Transform tangent data and return:
    return Xg.dot(R.T)    
    

#------------------------ Classes --------------------------------
class Manifold(object):
    def __init__(self,
                 n_dimM=None, n_dimT=None, 
                 exp_e=None, log_e=None, id_elem=None, 
                 name='', f_nptoman=None, 
                 f_mantonp= None, manlist=None,
                 f_action = None,
                 f_parallel_transport= None,
                 exp=None, log=None
                 ):
        '''
        To specify a 'root' manifold one needs to provide:
        n_dimM : Dimension of Manifold space
        n_dimT : Dimension of Tangent space
        exp    : Exponential function that maps elements from the tangent space to the manifold
        log    : Logarithmic function that maps elements from the manifold to the tangent space
        init   : (optional) an initialiation element
        And optionally:
        f_nptoman: A function that defines how a np array of numbers is tranformed into a manifold element
        If these parameters are not provides, some of the functions of the manifold don't work
        
        Alternatively one can create a composition of root manifolds by providing a list of manifolds
        '''
        
        # Check input
        if  ( ((n_dimM is None) and (n_dimT is None) and 
                (exp_e is None) and (log_e is None) and (id_elem is None) and
                (f_action is None)
              )
            and 
            (manlist is None)
             ):
            raise RuntimeError('Improper Manifold specification, either specify root manifold by providing atleast' +
                               'n_dimM, n_dimT, exp, log, init, fcocM, fcocT or provide a manifold list')
        
        if manlist is not None:
            # None-root manifold (i.e. a product of manifolds)
            self.__manlist = []
            self.n_dimT = 0
            self.n_dimM = 0
            id_list = [] # List of identity elements
            name = ''
            for i,man in enumerate(manlist):
                # Check existance of manifold and add to list
                if type(man) is Manifold:
                    self.__manlist.append(man)
                else:
                    raise RuntimeError('Non-manifold type found in manifold list.')
                    
                # Gather properties of cartesian product of manifolds:
                self.n_dimT += man.n_dimT
                self.n_dimM += man.n_dimM
                id_list.append(man.id_elem)
                # combine names:
                if i > 0:
                    name = '{0}, {1}'.format(name, man.name)
                else:
                    name = '{0}'.format(man.name)
            self.name = name

            # Assign functions to manifold:
            if len(manlist) > 1:
                # If there are more manifold in the list, assign the public manifold functions:
                self.id_elem = tuple(id_list)
                self.__fnptoman = self.np_to_manifold
                self.__fmantonp = self.manifold_to_np
                self.__fparalleltrans = self.parallel_transport
                self.__faction  = self.action

                # Mapping functions are defined recursively through the exp, and log functions of this manifold
                self.__fexp = self.exp
                self.__flog = self.log
            else:
                # If there is only one manifold in the list, we need to use the dedicated functions
                self.id_elem = manlist[-1].id_elem 
                self.__fnptoman = manlist[-1].np_to_manifold
                self.__fmantonp = manlist[-1].manifold_to_np
                self.__fparalleltrans = manlist[-1].parallel_transport
                self.__faction  = manlist[-1].action

                # Mapping functions are defined recursively through the exp, and log functions of this manifold
                self.__fexp = manlist[-1].exp
                self.__flog = manlist[-1].log

                        
            
        else:
            # Root manifold:
            if n_dimT is None:
                n_dimT = n_dimM
                
            if f_nptoman is None:
                # Specify the function as a simple one-to-one mapping:
                f_nptoman = lambda data: data #if type(data) is np.ndarray else np.array([data])
                self.__fnptoman = f_nptoman
            else:
                self.__fnptoman = f_nptoman

            if f_mantonp is None:
                # Specify the function as a simple one-to-one mapping:
                f_mantonp = lambda data: data #if type(data) is np.ndarray else np.array([data])
                self.__fmantonp = f_mantonp 
            else:
                self.__fmantonp = f_mantonp
                
            # Assign action functions:
            self.__faction = f_action
            self.__fparalleltrans = f_parallel_transport

            # Create the log and exp mappings using the base functions log and exp
            # Check if the log and exp maps are provided (this could yield faster computation)
            # Otherwise create them using the action function:
            if exp is None:
                self.__fexp = lambda x_tan, base, reg: f_action(exp_e(x_tan, reg), id_elem, base)
            else:
                self.__fexp = exp

            if log is None:
                self.__flog = lambda x    , base, reg: log_e( f_action(x,base, id_elem), reg)
            else:
                self.__flog = log


            # The manifold list only consists of the Manifold itself:            
            self.__manlist = [self]
            
            self.id_elem = id_elem
            self.n_dimT =n_dimT
            self.n_dimM =n_dimM
            self.name = name

        self.n_manifolds = len(self.__manlist)
        
    def __mul__(self,other):
        '''The product operator specifies the indirect product of the manifolds
        The implementation simply means 'stacking' the manifolds
        
        '''
        manlist = [] 
        for _,item in enumerate(self.__manlist):
            manlist.append(item)
        for _,item in enumerate(other.__manlist):
            manlist.append(item)
        
        return Manifold(manlist=manlist)
        
    def exp(self,g_tan, base=None, reg=1e-10):
        '''Convert element on Tangent space defined by base to element on Manifold
        base : tuple that represents the base of the tangent plane
        g_tan: n_data x n_dimT array
        '''
        # Single Manifolds will not have base in tuple form, adopt:
        if base is None:
            base = self.id_elem

        if type(base) is not tuple:
            base = tuple([base])

        # If g_tan only has a single dimension, we assume it consists of one data point
        d_added = False
        if g_tan.ndim==1:
            g_tan = g_tan[None,:]
            d_added=True;
            
        tmp = []
        ind = 0
        for i,man in enumerate(self.__manlist):
            g_tmp = man.__fexp(g_tan[:,np.arange( man.n_dimT ) + ind ], base[i], reg=1e-10 )
            if d_added:
                # Remove additional dimension
                tmp.append(g_tmp[0])
            else:
                tmp.append(g_tmp)
            ind += man.n_dimT
        
        if len(self.__manlist) ==1:
            return  tmp[0]
        else:
            return  tuple(tmp)
            
    def log(self,g, base=None, reg=1e-10):
        '''Convert element on Manifold to element on Tangent space defined by base'''
        # Single Manifolds will not have base in tuple form, adopt:
        if base is None:
            base = self.id_elem
        if type(g) is not tuple:
            g = tuple([g])
        if type(base) is not tuple:
            base = tuple([base])
        
        g_tan = []
        for i, man in enumerate(self.__manlist):
            g_tan.append( man.__flog(g[i], base[i], reg=reg) )

        return  np.hstack( g_tan )

    def action(self, X ,g, h):
        ''' Create manifold elements Y that have a relation with h and elements X have with g'''
        # Single Manifolds will not have base in tuple form, adopt:
        if type(g) is not tuple:
            g = tuple([g])
        if type(h) is not tuple:
            h = tuple([h])
        if type(X) is not tuple:
            X = tuple([X])
        
        # Perform actions
        Y = []
        for i, man in enumerate(self.__manlist):
            if man.__faction is None:
                raise RuntimeError('Action function not specified for manifold {0}'.format(man.name))
            else:
                Y.append( man.__faction(X[i], g[i], h[i]) )

        if len(self.__manlist) ==1:
            return  Y[0]
        else:
            return  tuple(Y)

    def parallel_transport(self, Xg, g, h, t=1):
        ''' Create manifold Parallel transport Xg from tangent space at g, to tangent space at h'''
        # Single Manifolds will not have base in tuple form, adopt:
        if type(g) is not tuple:
            g = tuple([g])
        if type(h) is not tuple:
            h = tuple([h])

        d_added =False;
        if Xg.ndim==1:
            Xg = Xg[None,:]
            d_added=True;
        
        # Perform actions
        Xh = np.zeros(Xg.shape)
        ind = 0
        for i, man in enumerate(self.__manlist):
            if man.__faction is None:
                raise RuntimeError('Action function not specified for manifold {0}'.format(man.name))
            else:
                sl = np.arange( man.n_dimT ) + ind
                Xh[:,sl] = man.__fparalleltrans(Xg[:,sl], g[i], h[i], t) 

            ind += man.n_dimT

        if d_added:
            return  Xh[0,:]
        else:
            return  Xh
    
    def np_to_manifold(self, data):
        '''Transfrom the nparray in a tuple of manifold elements
        data: n_data x n_dimM  array
        output: tuple( M1, M2, ..., Mn), in which M1 is a list of manifold elements
       
        '''
        tmp = []
        ind = 0
        for j, man in enumerate(self.__manlist):
            if data.ndim == 1:
                tmp.append(man.__fnptoman( data[np.arange(man.n_dimM) + ind] ) )
            else:
                tmp.append(man.__fnptoman( data[:,np.arange(man.n_dimM) + ind] ) )
            ind += man.n_dimM

        if j == 0:
            # Single manifold, remove the list structure:
            return  tmp[0] 
        else:
            # Combined manifold, transform the list in a tuple
            return tuple(tmp) 

    def manifold_to_np(self,data):
        # Ensure that we can handle both single samples and arrays of samples:
        np_list = []
        if len(self.__manlist) == 1:
            npdata = self.__fmantonp(data)
        else:
            for j, man in enumerate(self.__manlist):
                tmp = man.__fmantonp(data[j])
                #if (man.n_dimM == 1) and (tmp.ndim == 1):
                #    tmp = tmp[:,None]
                np_list.append(tmp)
            npdata = np.hstack(np_list)

        return npdata

    def swapto_tupleoflist(self,data):
        ''' Swap data from tuples of list to list of tuples'''
        if (type(data) is tuple) or (type(data) is np.ndarray):
            # Do nothing, already ok
            return data
        elif type(data) is list:
            # Data is list of individual tuples:
            #          1           ---          n_data
            #[ (  submanifold_1 )        (  submanifold_1 ) ]
            #[ (  |             ), ---  ,(  |             ) ]
            #[ (  submanifold_N )        (  submanifold_N ) ]
            # We swap to list:
            npdata = []
            for i,elem in enumerate(data):
                npdata.append( self.manifold_to_np(elem))
            npdata = np.vstack(npdata)
            tupledata = self.np_to_manifold(npdata)

            return tupledata
        else:
            raise RuntimeError('Unknown type {0} encoutered for swap'.format(type(data))) 

    def swapto_listoftuple(self,data):
        ''' Swap data from list of tuples to tuple of lists'''
        if type(data) is list or type(data) is np.ndarray:
            # Data already ok:
            return data
        if type(data) is tuple:
            # Data is tuple of lists
            # ( [nbdata x submanifold_1]  )
            # ( [ |                    ]  )
            # ( [nbdata x submanifold_N]  )
            
            npdata = []
            for i, elem in enumerate(list(data)):
                tmp = self.get_submanifold(i).manifold_to_np(elem)
                npdata.append(tmp)
            npdata = np.hstack(npdata)
            
            elemlist = []
            if npdata.ndim==2:
                for i in range(npdata.shape[0]):
                    elemlist.append( self.np_to_manifold(npdata[i,:]) )
            elif npdata.ndim==1:
                # Only one element
                elemlist.append( self.np_to_manifold(npdata) )
            else:
                raise RuntimeError('Cannot handle dimensionallity of Array')

            return elemlist
        else:
            raise RuntimeError('Unknown type {0} encoutered for swap'.format(type(data))) 

    def swap_btwn_tuplelist(self, data):
        ''' Swap between tuple of data points and list of tuples'''
        if type(data) is list:
            return self.swapto_tupleoflist(data)
        elif type(data) is tuple:
            return self.swapto_listoftuple(data)
        elif type(data) is np.ndarray:
            return data
        else:
            raise RuntimeError('Unknown type {0} encoutered for swap'.format(type(data))) 
    
    def get_submanifold(self, i_man):
        ''' Returns a manifold that of the requested indices
        i_man : (list of) manifold index
        
        '''

        if type(i_man) is list:
            # List of indices requested:
            manlist = []
            for _, ind in enumerate(i_man):
                manlist.append(self.get_submanifold(ind))
            #if len(manlist)==1:
            #   return manlist[-1] # Return single manifold
            #lse:
            return Manifold(manlist=manlist) # Return new combination of manifolds
        else:
            # Check input
            if i_man > len(self.__manlist):
                raise RuntimeError('index {0} exceeds number of submanifolds.'.format(i_man) )
            # Return requested sub-manifold
            return self.__manlist[i_man]

    def get_tangent_indices(self, i_man):
        '''Get the tangent space indices for a (list of) manifold(s)
        i_man : (list of) manifold index
        '''
        if type(i_man) is list:
            # List of indices requested:
            indlist = []
            for _, ind in enumerate(i_man):
                # Get manifold indices:
                tmp = self.get_tangent_indices(ind)

                # Copy indices:
                for _,i in enumerate(tmp):
                    indlist.append(i)
            return indlist
        else:
            # Check input
            if i_man > len(self.__manlist):
                raise RuntimeError('index exceeds number of submanifolds')
            
            # Create & return range
            st_in = 0;
            for i in range(0,i_man):
                st_in += self.__manlist[i].n_dimT
            return np.arange(st_in, st_in + self.__manlist[i_man].n_dimT ).tolist()

    def n_manifolds(self):
        return len(self.__manlist)

# Define two standard manifolds:
def get_euclidean_manifold(n_dim,name='Euclidean Manifold'):
    return Manifold(n_dimM=n_dim, n_dimT=n_dim,
                        exp_e=eucl_exp_e, log_e=eucl_log_e, id_elem=np.zeros(n_dim), 
                        name=name, 
                        f_action = eucl_action,
                        f_parallel_transport = eucl_parallel_transport
                        )

def get_quaternion_manifold(name='Quaternion Manifold'):
    return Manifold(n_dimM=4, n_dimT=3, 
                 exp_e=quat_exp_e, log_e=quat_log_e, id_elem=quat_id, 
                 name=name, 
                 f_nptoman= ar.Quaternion.from_nparray,
                 f_mantonp= ar.Quaternion.to_nparray_st,
                 f_action= quat_action,
                 f_parallel_transport = quat_parallel_transport,
                 exp=quat_exp, log=quat_log # Add optional non-base maps that to provide more efficient computation
                    )

def get_s2_manifold(name='S2', fnptoman=None, fmantonp=None):
    return Manifold(n_dimM=3, n_dimT=2, 
                 exp_e=s2_exp_e, log_e=s2_log_e, id_elem=s2_id, 
                 name=name, 
                 f_nptoman= fnptoman,
                 f_mantonp= fmantonp,
                 f_action=s2_action,
                 f_parallel_transport=s2_parallel_transport,
                 exp=s2_exp, log=s2_log# Add optional non-base maps that to provide more efficient computation
                 )
