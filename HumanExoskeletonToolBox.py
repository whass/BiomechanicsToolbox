"""
    Author : Walid Hassani
    This toolbox is developped for my PhD "Contribution to the modeling and he control of wearable knee joint
    exoskeleton" at LISSI Lab (University Paris-Est)
"""

from numpy.ma import sin, cos


class HumanExoskeletonModel:
    def __init__(self, inertia, damping, stiffness, gravitationnalTorque):
        self.inertia = inertia
        self.damping = damping
        self.stiffness = stiffness
        self.gravitationnalTorque = gravitationnalTorque
        self.xDot = []
        for i in range(0, 1):
            self.xDot.append(0)

    def FullSingleJointModel(self, exoskeletonGeneratedTorque, humanGeneratedTorque, position, velocity, restPosition):
        self.exoskeletonGeneratedTorque = exoskeletonGeneratedTorque
        self.humanGeneratedTorque = humanGeneratedTorque
        self.position = position
        self.velocity = velocity
        self.restPosition = restPosition
        self.acceleration = 1 / self.inertia * (self.exoskeletonGeneratedTorque + self.humanGeneratedTorque - (
            self.damping * self.velocity + self.stiffness*(
                self.position - self.restPosition) + self.gravitationnalTorque * sin(
                self.position - self.restPosition)))
        return self.acceleration

    def LinearizedSingleJointModel(self, exoskeletonGeneratedTorque, humanGeneratedTorque, position, velocity, restPosition, referencePosition):
        self.exoskeletonGeneratedTorque = exoskeletonGeneratedTorque
        self.humanGeneratedTorque = humanGeneratedTorque
        self.position = position
        self.referencePosition = referencePosition
        self.velocity = velocity
        self.restPosition = restPosition
        self.acceleration = 1 / self.inertia * (self.exoskeletonGeneratedTorque + self.humanGeneratedTorque - (
            self.damping * self.velocity + self.stiffness*(
                self.position - self.restPosition) + self.gravitationnalTorque * cos(self.referencePosition - self.restPosition)*(self.referencePosition - self.restPosition)))
        return self.acceleration


