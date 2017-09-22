'''
This code file serves as a minimal example rosnode for GMM-GMR based on riepybdlib. It assumes the regression from 1-dimensional Eucliden space to robot pose (R^3 x S^3). Other types of regression will require modification of the GMRNode class.

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

# ROS related:
import rospy
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger, TriggerResponse

# Riepybdlib stuff:
import riepybdlib.statistics as rs
import riepybdlib.manifold as rm

# Define manifolds:
m_e1 = rm.get_euclidean_manifold(1)
m_e3 = rm.get_euclidean_manifold(3)
m_q  = rm.get_quaternion_manifold()
m_e1xe3xq  = m_e1*m_e3*m_q


def pose_from_tuple(posetuple):
    '''Convert tuple of position and quaternion to ROS Pose msg'''
    pose = Pose()
    pose.position.x = posetuple[0][0]
    pose.position.y = posetuple[0][1]
    pose.position.z = posetuple[0][2]

    pose.orientation.w = posetuple[1].q0
    pose.orientation.x = posetuple[1].q[0]
    pose.orientation.y = posetuple[1].q[1]
    pose.orientation.z = posetuple[1].q[2]

    return pose


class GMRNode(object):

    def __init__(self, modelname, n_components):
        ''' Initialize node
        modelname   : base name of GMM model
        n_components:  Number of states required to load GMM model
        '''

        # Define publishers & Service
        self.pub_gmrout = rospy.Publisher('GMR_out', Pose, queue_size=1)     # Node to publish outcome of GMR  
        self.gmr_service = rospy.Service('/pbd/doGMR', Trigger, self._doGMR) # Make GMR service available

        # load model:
        self._gmm = rs.GMM.load(modelname, n_components, m_e1xe3xq)   

    def _doGMR(self, call):
        ''' Call back for GMR service'''

        print('Received call for GMR service')

        # Handle timing:
        tmax = self._gmm.gaussians[-1].mu[0] # Some final time 
        freq = 200.0  # Sampling frequency

        # GMR loop:
        t=0
        myrate = rospy.Rate(freq)  
        while (t < tmax) and not rospy.is_shutdown(): # <- maybe add some cancellation policy

            despose = self._gmm.gmr(np.array([t]), 0,[1,2])[0].mu  # GMR from t-> pose
            rospose = pose_from_tuple(despose)  # Need some conversion function 
            self.pub_gmrout.publish(rospose)    # Publish current gmr res

            # Handle loop exit:
            t+= (1/freq)                        # Update time counter
            myrate.sleep()                      # Keep fixed rate

        return TriggerResponse(True, "")


if __name__=="__main__":
    # Set model parameters:
    modelname = '...'  #<-Insert your model name here
    n_components = 6   #<-Insert number of states of the model

    # Create node:
    rospy.init_node("Online_GMR")             # init node
    mygmr = GMRNode(modelname, n_components)  # Create node
    print('GMR node started')

    rospy.spin()                              # Prevent close



