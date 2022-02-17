import time
import math
import numpy as np



class PIDController:

    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.past_error = 0.0
        self.prev_time = 0.0

    def output(self, goal, current=0):
        cur_time = time.time()
        dt = cur_time - self.prev_time

        lat_error = goal - current

        proportional = self.Kp * lat_error
        self.integral += self.Ki * (lat_error + self.past_error) * dt
        derivative = self.Kd * (lat_error - self.past_error) / dt

        self.integral = min(self.integral, 1000)

        output = proportional + self.integral + derivative

        self.past_error = lat_error
        self.prev_time = cur_time

        return output