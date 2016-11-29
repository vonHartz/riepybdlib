import numpy as np
import sympy as sp


########################### EULER ANGLES #############################################################
# Define basic rotations:
# We use the conventions r = R*r'
# i.e. a unit rotation defines a rotation from the body fixed coordinates to the world-frame coordinates


def fRx(theta):

    return sp.Matrix([[1, 0       , 0           ],
                      [0, sp.cos(theta),-sp.sin(theta)],
                      [0, sp.sin(theta), sp.cos(theta)]
                     ])
def fRy(theta):
    return sp.Matrix([[sp.cos(theta), 0 , sp.sin(theta)],
                      [0            , 1 , 0 ],
                      [-sp.sin(theta), 0 , sp.cos(theta)]
                     ])
    
def fRz(theta):
    return sp.Matrix([[sp.cos(theta), -sp.sin(theta)  , 0],
                      [sp.sin(theta), sp.cos(theta) , 0],
                      [0            , 0           , 1]
                    ])

R_dict = {'x': fRx,'y': fRy, 'z': fRz}
ax_dict = {'x': 0  ,'y': 1  , 'z': 2}

def getEulerR(anglespec):
    R = sp.eye(3)
    for ind,(ax,symbol) in (enumerate(anglespec)):
        R *= R_dict[ax](symbol)
    return R

def EulerToBodyW(anglespec):
    ''' function te create the mapping b between Euler Angles 
    and Body Fixed Angular velocities, i.e.
      w' = b*angles
    
    where w are the angular velocities and angles is a list of euler angles
    
    '''
    # dictionary with rotation functions for different axis
    
    B = sp.zeros( 3,len(anglespec) ) 
    R = sp.eye(3)                     
    
    for ind,(ax,symbol) in reversed(list(enumerate(anglespec))):
        # index selector:
        tmp = sp.zeros( 3, len(anglespec))
        tmp[ ax_dict[ax], ind] = 1
        
        B += R*tmp # Stack transformation
        
        # Stack Rotation for next iteration:
        R *= R_dict[ax](symbol).T
        
    return B


def EulerToWorldW(anglespec):
    ''' function te create the mapping b between Euler Angles 
    and World Fixed Angular velocities, i.e.
      w = b*angles
    
    where w are the angular velocities and angles is a list of euler angles
    
    '''
    # dictionary with rotation functions for different axis
    Rdict = {'x': fRx,'y': fRy, 'z': fRz}
    Adict = {'x': 0  ,'y': 1  , 'z': 2}
    
    B = sp.zeros( 3,len(anglespec) )
    R = sp.eye(3)
    
    n = len(anglespec)
    for ind,(ax,symbol) in enumerate(anglespec):
        # index selector:
        tmp = sp.zeros(3, len(anglespec))
        tmp[Adict[ax],ind] = 1
        
        # add to transformation
        B += R*tmp
        
        # compute rotation for next iteration:
        R *= Rdict[ax](symbol)
        
    return B


################################### QUATERNIONS 


def skew(q):
    return np.array([ [ 0   ,-q[2], q[1] ], 
                      [ q[2], 0   ,-q[0] ],
                      [-q[1], q[0], 0    ] ])

class Quaternion(object):
    def __init__(self,q0,q):
        '''
        a quaterion consist of a scalar q0 and a three dimensional vector q
        '''
        if (len(q) == 3):
            self.q0 = float(q0) 
            self.q  = q 
        else:
            print('.q.shape {0}'.format(q.shape))
            raise TypeError
        
    def __neg__(self):
        return Quaternion(-self.q0,-self.q)
    
    def __add__(self,other):
        '''
        q = self
        p = other
        q + p = (q0 + p0, qvec + pvec)
        '''

        if type(other) is Quaternion:
            return Quaternion(self.q0 + other.q0,self.q+other.q) 
        elif type(other) is list:
            qlist = []
            for _,q in enumerate(other):
                qlist.append(self+q)
            return qlist
    
    def __sub__(self,other):
        '''
        q = self
        p = other
        q - p = (q0 - p0, qvec - pvec)
        '''
        if type(other) is Quaternion:
            return Quaternion(self.q0 - other.q0, self.q - other.q)
        elif type(other) is list:
            qlist = []
            for _,q in enumerate(other):
                qlist.append(self-q)
            return qlist

        
        
    def __mul__(self,other):
        ''' quaternion product (non-commutative):
            q = self
            p = other
            'x' indicates the cross product
            q*p = (q0*p0 - qvec*pvec, q0*pvec + p0*qvec + qvec x pvec)
        
        '''

        if type(other) is Quaternion:
            v0 = self.q0*other.q0 - self.q.dot(other.q)
            v  = self.q0*other.q + other.q0*self.q + np.cross(self.q,other.q)
            return Quaternion(v0,v)
        elif type(other) is list:
            qlist = []
            for _,q in enumerate(other):
                qlist.append(self*q)
            return qlist

    
    def adj(self):
        '''
        The adjoint of the quaternion (q0, -qvec)
        '''
        return Quaternion(self.q0,-self.q)
    
    def norm(self):
        #return np.sqrt( (self.adjoint() * self).q0 )
        return np.sqrt(self.q0**2 + self.q.dot(self.q))
        
    def Q(self):
        '''
        Quaternion matrix
        '''
        Q1   = np.hstack(( self.q0, -self.q ) )
        Q234 = np.hstack( (self.q[:,None] , self.q0*np.eye(3) + skew(self.q) )) 
        Q = np.vstack((Q1,Q234))
        
        return Q
    def to_nparray(self):
        if type(self) is Quaternion:
            return np.hstack( ([self.q0],self.q))

        elif type(self) is list:
            qarray = np.zeros( (len(self), 4) )
            for i,q in enumerate(self):
                qarray[i,:] = q.to_nparray()
            return qarray
        else:
            raise RuntimeError('Argument is is of invalid type {0}'.format(type(self)))

    @staticmethod
    def from_nparray(qarray):
        ''' Return list of Quaternions from an np array
        qarray: n_data x 4 numpy array in which each column is [q0, q1, q2, q3]
        '''
        if qarray.ndim==1:
            # Single sample:
            return Quaternion(qarray[0], qarray[1:])
        else:
            qlist = []
            for i in range(qarray.shape[0]):
                qlist.append(Quaternion(qarray[i,0], qarray[i,1:]))
            return qlist


    def adjQ(self):
        '''
        Quaternion matrix
        '''
        Q1   = np.hstack(( self.q0, -self.q ) )
        Q234 = np.hstack( (self.q[:,None] , self.q0*np.eye(3) - skew(self.q) )) 
        Q = np.vstack((Q1,Q234))
        
        return Q
    
    
    def normalized(self):
        norm = self.norm()
        return Quaternion(self.q0/norm, self.q/norm)
    
    
    def i(self):
        ''' Reciprocal (inverse) of a Quaternion'''
        qbar = self.adj()
        norm2 = self.norm()**2
        return Quaternion(qbar.q0/norm2, qbar.q/norm2)
    
    
    def R(self):
        ''' From Peter Corke's Matlab robotics toolbox'''
        s = self.q0
        x = self.q[0]
        y = self.q[1]
        z = self.q[2]

        R = np.array([
            [1-2*(y**2+z**2), 2*(x*y-s*z)    , 2*(x*z+s*y)],
            [2*(x*y+s*z)    , 1-2*(x**2+z**2), 2*(y*z-s*x)],
            [2*(x*z-s*y)    , 2*(y*z+s*x)    , 1-2*(x**2+y**2)]
            ])
        return R
    
    def __str__(self):
        return "({0:.2f}, {1})".format(self.q0,self.q) 

############## Rotational Conversions:

def quatToEulerXYZ(q):
    q = q.normalized()
    R = q.R();
    
    # NASA paper:
    th1 = np.arctan2(-R[1,2],R[2,2])
    th2 = np.arctan2(R[0,2],np.sqrt(1-R[0,2]**2))
    th3 = np.arctan2(-R[0,1],R[0,0])
    
    
    return np.array([th1,th2,th3])


def get_q_from_R(R):
    ''' From Peter Corke's Robotics toolbox'''

    qs = np.sqrt( np.trace(R) +1)/2.0
    kx = R[2,1] - R[1,2]   # Oz - Ay
    ky = R[0,2] - R[2,0]   # Ax - Nz
    kz = R[1,0] - R[0,1]   # Ny - Ox

    if (R[0,0] >= R[1,1]) and (R[0,0] >= R[2,2]) :
        kx1 = R[0,0] - R[1,1] - R[2,2] + 1 # Nx - Oy - Az + 1
        ky1 = R[1,0] + R[0,1]              # Ny + Ox
        kz1 = R[2,0] + R[0,2]              # Nz + Ax
        add = (kx >= 0)
    elif (R[1,1] >= R[2,2]):
        kx1 = R[1,0] + R[0,1]              # Ny + Ox
        ky1 = R[0,0] - R[1,1] - R[2,2] + 1 # Oy - Nx - Az + 1
        kz1 = R[2,1] + R[1,2]              # Oz + Ay
        add = (ky >= 0)
    else:
        kx1 = R[2,0] + R[0,2]              # Nz + Ax
        ky1 = R[2,1] + R[1,2]              # Oz + Ay
        kz1 = R[2,2] - R[0,0] - R[1,1] + 1 # Az - Nx - Oy + 1
        add = (kz >= 0)

    if add:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1

    nm = np.linalg.norm(np.array([kx, ky, kz]))
    if nm == 0:
        q = Quaternion(1, np.zeros(3))
    else:
        s = np.sqrt(1 - qs**2) / nm
        qv = s*np.array([kx, ky, kz])
        q = Quaternion(qs, qv)

    return q


#### Axis-Angle Representation:
def get_axisangle(d, e=None, reg=1e-6):
    ''' Compute axis angle of axis de w.r.t e
        d: Vector [1d ndarray]
        e: Vector [1d ndarray]
        
        Computation:
        ax    = (e x d) / (|| e x d ||)
        angle = arccos( d * e ) 

        * is in product
        x is the cross product

        if no e is provided, we adopt: e = [0, 0, 1]^T

    '''

    if e is None:
        e = np.array([0,0,1])
        norm = np.sqrt(d[0]**2 + d[1]**2)
        if norm < reg:
            return (e,0)
        else:
            vec = np.array( [-d[1], d[0], 0])
            return vec/norm, np.arccos(d[2])
    else:
        # Compute cross product between identity and d
        exd = skew(e).dot(d)
        norm = np.linalg.norm(exd)

        # Check norm:        
        if norm < reg:
            # smaller than reguralization, assume no rotation:
            ax = e
            angle = 0
        else:
            # Rotation is present:
            ax = exd/norm
            angle = np.arccos( (d*e).sum( axis=(d.ndim-1) ) )
        return (ax, angle)


def R_from_axis_angle(ax, angle):
    ''' Get Rotation matrix from axis angle representation using Rodriguez formula
        ax   : The unit axis defining the axis of rotation [ 1d ndarray]
        angle: Angle of rotation [float]

        Return: 
        R(ax, angle) = I + sin(angle) x ax + (1 - cos(angle) ) x ax^2

        where x is the cross product
    '''

    utilde = skew(ax)
    return np.eye(3) + np.sin(angle)*utilde + (1 - np.cos(angle))*utilde.dot(utilde)
