import math 
import time 
import numpy as np


class StanleyController:
    def __init__(self, k_e=0.0, k_s=0.0, vel=0.0):

        self.k_e = k_e
        self.k_s = k_s
        self.vel = vel
        self._cross_track_error = 0.0
        self.prev_yaw = 0.0
        self.steering = 0.0


    def output(self, goal, current=0):

        
        self._cross_track_error = goal - current
        
        _cross_track_steering = self.cross_track_steering(self._cross_track_error, self.k_e, self.vel, self.k_s)
        #_heading_error = self.heading_error(self.slope, self.prev_yaw)


        #steering = _heading_error + _cross_track_steering
        self.steering = self.steering + _cross_track_steering

        
        
        if (self.steering > np.pi):
            self.steering -= 2*np.pi

        if (self.steering < -np.pi):
            self.steering += 2*np.pi
        
        

        return self.steering

    def cross_track_steering(self, _cross_track_error, k_e, vel, k_s):

        return (math.atan((self.k_e * self._cross_track_error) / (k_s + self.vel)))

    """
    def heading_error(self, self.slope, self.prev_yaw):

        _heading_error = self.slope - self.prev_yaw

        if (_heading_error > np.pi):
            _heading_error -= 2*np.pi

        if (_heading_error < -np.pi):
            _heading_error += 2*np.pi

        self.prew_yaw = self.slope

        return _heading_error
    """
 