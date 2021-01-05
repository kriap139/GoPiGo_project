import numpy as np
import time

__all__ = ['PIDController']


class PIDController:
    def __init__(self, kp, kd, maxAngularSpeed, initSearchSpeed=100, minSearchSpeed=30, searchDecrement=0.5,
                 searchRadius=9, radiusIncrement=1):

        self.kp = kp
        self.kd = kd
        self.maxAngularSpeed = maxAngularSpeed
        self.maxAngularSpeedNeg = -1 * maxAngularSpeed

        self.initSearchSpeed = initSearchSpeed
        self.minSearchSpeed = minSearchSpeed
        self.searchDecrement = searchDecrement
        self.searchRadius = searchRadius
        self.radiusIncrement = radiusIncrement

        self.searchSpeed = initSearchSpeed

    def getAngularSpeed(self, lateralError, angularError):
        self.searchSpeed = self.initSearchSpeed

        p_term = -self.kp * lateralError
        d_term = -self.kp * self.kd * np.sin(angularError)

        angularSpeed = p_term + d_term

        return self.checkAngularSpeed(angularSpeed), (p_term, d_term)

    def checkAngularSpeed(self, angularSpeed):
        if angularSpeed > self.maxAngularSpeed:
            return self.maxAngularSpeed
        elif angularSpeed < self.maxAngularSpeedNeg:
            return self.maxAngularSpeedNeg
        else:
            return angularSpeed

    def getSearchSpeed(self):
        self.searchSpeed -= self.searchDecrement

        if self.searchSpeed <= self.minSearchSpeed:
            return self.minSearchSpeed
        else:
            return self.searchSpeed
