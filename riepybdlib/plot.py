'''
Writing code takes time. Polishing it and making it available to others takes longer! 
If some parts of the code were useful for your research of for a better understanding 
of the algorithms, please reward the authors by citing the related publications, 
and consider making your own research available in this way.

@article{Zeestraten2017,
  	title = {An Approach for Imitation Learning on Riemannian Manifolds},
	author = {Zeestraten, M.J.A. and Havoutis, I. and Silverio, J. and Calinon, S. and Caldwell, D. G.},
	journal={{IEEE} Robotics and Automation Letters ({RA-L})},
	year = {2017},
	month={January},
}

 
Copyright (c) 2017 Istituto Italiano di Tecnologia, http://iit.it/
Written by Martijn Zeestraten, http://www.martijnzeestraten.nl/

This file is part of RiePybDlib, http://gitlab.martijnzeestraten.nl/martijn/riepybdlib

RiePybDlib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.
 
RiePybDlib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RiePybDlib. If not, see <http://www.gnu.org/licenses/>.
'''


import numpy as np
import scipy as sp
import scipy.linalg 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
import matplotlib.cm as cm

import riepybdlib.angular_representations as ar

def fRx(theta):

    return np.array([[1, 0       , 0           ],
                      [0, np.cos(theta),-np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]
                     ])
def fRy(theta):
    return np.array([[np.cos(theta), 0 , np.sin(theta)],
                      [0            , 1 , 0 ],
                      [-np.sin(theta), 0 , np.cos(theta)]
                     ])
    
def fRz(theta):
    return np.array([[np.cos(theta), -np.sin(theta)  , 0],
                      [np.sin(theta), np.cos(theta) , 0],
                      [0            , 0           , 1]
                    ])


def plotRotation(ax, q, pos=np.zeros(3), length=1, alpha=1,color=None,label='', **kwargs):
    R = q.R();
        
    cols = np.eye(3)
        
    for i in range(3):
        xs = [0,R[0,i]*length] + pos[0]
        ys = [0,R[1,i]*length] + pos[1]
        zs = [0,R[2,i]*length] + pos[2]
        if i==1:
            label='' # Reset label to only let it appear once
        if color is None:
            ax.plot(xs=xs, ys=ys, zs=zs, color=cols[i,], label=label, alpha=alpha,**kwargs)
        else:
            ax.plot(xs=xs, ys=ys, zs=zs, color=color, label=label,alpha=alpha,**kwargs)

def plotquatCov(ax, q, sigma, pos=np.zeros(3), axlength=1, covscale=1,alpha=1,linewidth=1):
    '''Plot rotation covariance '''
    cols = np.eye(3)
    Raxis = q.R()

    # Plot axis:
    for i in range(3):
        xs = [0,Raxis[0,i]*axlength] + pos[0]
        ys = [0,Raxis[1,i]*axlength] + pos[1]
        zs = [0,Raxis[2,i]*axlength] + pos[2]
        plt.plot(xs=xs, ys=ys, zs=zs, color=cols[i,], alpha=alpha,
                linewidth=3)
    
    
    # Plot Covariance:
    n_drawingsegments = 30
    t = np.linspace(-np.pi, np.pi, n_drawingsegments)
    
    R = sp.linalg.sqrtm(sigma*covscale)
    for a in range(3):
        if a==0:
            Rx = fRx(np.pi/2)
            tmp = np.vstack([np.zeros(n_drawingsegments),
                      np.cos(t),
                      np.sin(t)]).T
            eo = tmp.dot( (Rx.dot(R)).T)
        elif a==1:
            Ry = fRy(np.pi/2)
            tmp = np.vstack([
                    np.cos(t),
                    np.zeros(n_drawingsegments),
                    np.sin(t)]).T
            eo = tmp.dot( (Ry.dot(R)).T )
        elif a==2:
            Rz = fRz(np.pi/2)
            tmp = np.vstack([
                      np.cos(t),
                      np.sin(t),
                      np.zeros(n_drawingsegments)
                      ]).T
            eo = tmp.dot( (Rz.dot(R)).T )
        
        # Generate points for covariance
        points = eo.dot(Raxis.T)+ Raxis[:,a]*axlength + pos
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        
        vertices = [[i for i in range(len(x))]]
        tupleList = list(zip(x, y, z))

        poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
        collection = Poly3DCollection(poly3d, linewidths=1, alpha=0.2)
        collection.set_facecolor(cols[a,])

        ax.add_collection3d(collection)
        ax.plot(x,y,z,color=cols[a,],linewidth=linewidth)
        ax.plot(points[:,0], points[:,1], points[:,2],
               color=cols[a,], linewidth=linewidth)



def periodic_clip(val,n_min,n_max):
    ''' keeps val within the range [n_min, n_max) by assuming that val is a periodic value'''
    if val<n_max and val >=n_min:
        val = val
    elif val>=n_max:
        val = val - (n_max-n_min)
    elif val<n_max:
        val = val + (n_max-n_min)
    
    return val
        
        
def tri_elipsoid(n_rings,n_points):
    ''' Compute the set of triangles that covers a full elipsoid of n_rings with n_points per ring'''
    tri = []
    for n in range(n_points-1):
        # Triange down
        #       *    ring i+1
        #     / |
        #    *--*    ring i
        tri_up = np.array([n,periodic_clip(n+1,0,n_points),
                          periodic_clip(n+n_points+1,0,2*n_points)])
        # Triangle up
        #    *--*      ring i+1
        #    | / 
        #    *    ring i
        
        tri_down = np.array([n,periodic_clip(n+n_points+1,0,2*n_points),
                          periodic_clip(n+n_points,0,2*n_points)])
        
        tri.append(tri_up)
        tri.append(tri_down)
        
    tri = np.array(tri)
    trigrid = tri
    for i in range(1,n_rings-1):
        trigrid = np.vstack((trigrid,tri+n_points*i))
  
    return np.array(trigrid)

def plot_gaussian_2d(mu, sigma, ax=None, 
        linewidth=1, alpha=0.5, color=[0.6,0,0], label='',**kwargs):
    ''' This function displays the parameters of a Gaussian .'''

    # Create axis if not specified
    if ax is None:
        ax = plt.gca();

    nbDrawingSeg = 35;    
    t     = np.linspace(-np.pi, np.pi, nbDrawingSeg);    

    # Create Polygon
    #polyargs = {key: value for key, value in kwargs.items()
    #        if key in plt.Polygon.co_varnames}

    R = np.real(sp.linalg.sqrtm(1.0*sigma))
    points = (R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2,nbDrawingSeg])).T + mu).T
    polygon = plt.Polygon(points.transpose().tolist(),facecolor=color,alpha=alpha,linewidth=linewidth
            )
    ax.add_patch(polygon)                     # Patch

    pltargs = {key: value for key, value in kwargs.items()
            if key in ax.co_varnames}
    l,= ax.plot(mu[0], mu[1], '.', color=color, label=label, **pltargs) # Mean
    ax.plot(points[0,:], points[1,:], color=color, linewidth=linewidth, markersize=2,**pltargs) # Contour

    return l

def plot_gaussian_3d(mu, sigma, ax=None, n_points=30, n_rings=20, 
        linewidth=0, alpha=0.5, color=[0.6,0,0], label='', **kwargs):
    ''' Plot 3d Gaussian'''

    # Create axis if not provided
    if ax is None:
        ax = plt.gca();

    # Compute eigen components:
    (D0,V0) = np.linalg.eig(sigma)
    U0 = np.real(V0.dot(np.diag(D0)**0.5))
     
    # Compute first rotational path
    psi = np.linspace(0,np.pi*2,n_rings,endpoint=True)
    ringpts = np.vstack((np.zeros((1,len(psi))),np.cos(psi),np.sin(psi)))
    
    U = np.zeros((3,3))
    U[:,1:3] = U0[:,1:3]
    ringtmp = U.dot(ringpts)
    
    # Compute touching circular paths
    phi   = np.linspace(0,np.pi,n_points)
    pts = np.vstack((np.cos(phi),np.sin(phi),np.zeros((1,len(phi)))))
    
    xring = np.zeros((n_rings,n_points,3))
    for j in range(n_rings):
        U = np.zeros((3,3))
        U[:,0] = U0[:,0]
        U[:,1] = ringtmp[:,j]
        xring[j,:] = (U.dot(pts).T + mu)
        
    # Reshape points in 2 dimensional array:
    points = xring.reshape((n_rings*n_points,3))
          
    # Compute triangle points:
    triangles = tri_elipsoid(n_rings,n_points)
    
    # Plot surface:
    ax.plot_trisurf(points[:,0],points[:,1],points[:,2],
                   triangles=triangles,linewidth=linewidth,alpha=alpha,color=color,edgecolor=color)


############## S2 plot functions:


def get_axisangle(d):
    norm = np.sqrt(d[0]**2 + d[1]**2)
    if norm < 1e-6:
        return (np.array([0,0,1]),0)
    else:
        vec = np.array( [-d[1], d[0], 0 ])
        return ( vec/norm,np.arccos(d[2]) )

def plot_s2(ax,base=[0,0,1],color=[0.8,0.8,0.8],alpha=0.8,r=0.99, linewidth=0, **kwargs):

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
    ax.plot(xs=[base[0]], ys=[base[1]], zs=[base[2]],marker='*',color=color)


def plot_tangentplane_s2(ax, base, l_vert=1,color='gray',alpha=0.1, linewidth=0, **kwargs):
    # Tangent axis at 0 rotation:
    T0 = np.array([[1,0],
                  [0,1],
                  [0,0]])
    
    # Rotation matrix with respect to zero:
    (axis,ang) = get_axisangle(base)
    R = ar.R_from_axis_angle(axis, -ang)
    
    # Tangent axis in new plane:
    T = R.T.dot(T0)
    
    # Compute vertices of tangent plane at g
    hl = 0.5*l_vert
    X = [[hl,hl],  # p0
         [hl,-hl], # p1
         [-hl,hl], # p2
         [-hl,-hl]]# p3
    X = np.array(X).T
    points = (T.dot(X).T + base).T
    psurf = points.reshape( (-1,2,2))
    
    pltargs = {key: value for key, value in kwargs.items()
            if key in pybdplt.plot_surface.co_varnames}
    
    ax.plot_surface(psurf[0,:],psurf[1,],psurf[2,:],
                    color=color,alpha=alpha,linewidth=0, **pltargs)

def plot_gaussian_s2(ax,mu,sigma, color='red',linewidth=2, linealpha=1,planealpha=0.2,
                        label='', showtangent=True, **kwargs):
    
    # Plot Gaussian
    # - Generate Points @ Identity:
    nbDrawingSeg = 35;    
    t     = np.linspace(-np.pi, np.pi, nbDrawingSeg); 
    R = np.eye(3)
    R[0:2,0:2] = np.real(sp.linalg.sqrtm(1.0*sigma)) # Rotation for covariance
    (axis,angle) = get_axisangle(mu)
    R = ar.R_from_axis_angle(axis,angle).dot(R)      # Rotation for manifold location
    
    points = np.vstack( (np.cos(t), np.sin(t),np.ones(nbDrawingSeg)) )
    points = R.dot(points) 
    
    pltargs = {key: value for key, value in kwargs.items()
            if key in pybdplt.plot.co_varnames}
    
    l,= ax.plot(xs=mu[0,None], ys=mu[1,None], zs=mu[2,None], marker='.', 
            color=color,alpha=linealpha, label=label, **pltargs) # Mean

    ax.plot(xs =points[0,:], ys=points[1,:], zs=points[2,:], 
            color=color, 
            linewidth=linewidth, 
            markersize=2, alpha=linealpha,**pltargs) # Contour
    
    if showtangent:
        plot_tangentplane_s2(ax,mu,l_vert=1,color=color,alpha=planealpha, **kwargs)


def computeCorrelationMatrix(sigma):
    var = np.sqrt(np.diag(sigma))
    return  sigma/var[None,:].T.dot(var[None,:])

def plotCorrelationMatrix(sigma,labels=None,ax=None,labelsize=20):
    cormatrix = computeCorrelationMatrix(sigma)
    n_var = sigma.shape[0]
    if ax==None:
        plt.figure(figsize=(4,3))
        ax = plt.gca();

    if labels is None:
        labels = range(1,n_var+1)
    h = ax.pcolor(cormatrix, cmap='RdBu',vmax=1,vmin=-1)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(0,n_var)+0.5)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0,n_var)+0.5)
    ax.set_yticklabels(labels)
    ax.tick_params(labelsize=labelsize)
    l = plt.colorbar(h,ticks=[-1,0,1]);            
    l.ax.set_yticklabels([r'$-1$',r'$0$',r'$1$'])
    l.ax.tick_params(labelsize=labelsize)
       
