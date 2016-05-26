"""
    Author : Walid Hassani
    This toolbox is developped for my PhD "Contribution to the modeling and he control of wearable knee joint
    exoskeleton" at LISSI Lab (University Paris-Est)
"""
from numpy.ma import exp, sin
import numpy as np
from scipy import signal


class AnatomicalMuscleModel(object):
    """
        Compute muscle length and moment arm
    """

    def __init__(self, muscleName):
        print("Instantiation of the muscle :" + " " + muscleName)
        self.muscleLengthParam = []
        for i in range(5):
            self.muscleLengthParam.append(1)
        self.muscleMomentArmParam = []
        for i in range(4):
            self.muscleMomentArmParam.append(1)
        self.muscleParams = []
        for i in range(4):
            self.muscleParams.append(1)
        self.kneeAngle = 0
        self.hipAngle = 0
        self.muscleName = muscleName
        self.muscleLengthScale = 0
        if muscleName == "rf":
            self.muscleParams = ([0.084, 0.346, 1780, 5 * 3.14 / 180])
            self.muscleOptimalLength = self.muscleParams[0]
            self.tendonSlackLength = self.muscleParams[1]
            self.muscleMaximalIsometricForce = self.muscleParams[2]
            self.musclePennationAngle = self.muscleParams[3]
            self.muscleLengthParam = ([0.0044304, -0.0024913, -0.046268, 0.44897, -0.0434])
            self.muscleMomentArmParam = ([-0.011185, -0.047117, -0.032102, 0.044659])

        elif muscleName == "vl":
            self.muscleParams = ([0.084, 0.157, 1870, 5 * 3.14 / 180])
            self.muscleOptimalLength = self.muscleParams[0]
            self.tendonSlackLength = self.muscleParams[1]
            self.muscleMaximalIsometricForce = self.muscleParams[2]
            self.musclePennationAngle = self.muscleParams[3]
            self.muscleLengthParam = ([0.0004555, -0.0066869, -0.051383, 0.19889, 0])
            self.muscleMomentArmParam = ([-0.011185, -0.047117, -0.032102, 0.044659])

        elif muscleName == "vm":
            self.muscleParams = ([0.089, 0.126, 1295, 5 * 3.14 / 180])
            self.muscleOptimalLength = self.muscleParams[0]
            self.tendonSlackLength = self.muscleParams[1]
            self.muscleMaximalIsometricForce = self.muscleParams[2]
            self.musclePennationAngle = self.muscleParams[3]
            self.muscleLengthParam = ([8.6453 ** -5, -0.0068273, -0.050411, 0.17151, 0])
            self.muscleMomentArmParam = ([-0.011185, -0.047117, -0.032102, 0.044659])

        else:# choice == "vi"
            self.muscleParams = ([0.087, 0.136, 1235, 5 * 3.14 / 180])
            self.muscleOptimalLength = self.muscleParams[0]
            self.tendonSlackLength = self.muscleParams[1]
            self.muscleMaximalIsometricForce = self.muscleParams[2]
            self.musclePennationAngle = self.muscleParams[3]
            self.muscleLengthParam = ([0.0014072, -0.003718, -0.04929, 0.18153, 0])
            self.muscleMomentArmParam = ([-0.011185, -0.047117, -0.032102, 0.044659])

    def kneeJointMuscleLength(self, kneeAngle, hipAngle):
        self.kneeAngle = kneeAngle
        self.hipAngle = hipAngle
        self.muscleLength = self.muscleLengthParam[0] * self.kneeAngle ** 3 + self.muscleLengthParam[
            1] * self.kneeAngle ** 2 + self.muscleLengthParam[2] * self.kneeAngle + self.muscleLengthParam[3] + \
                       self.muscleLengthParam[4] * self.hipAngle
        return self.muscleLength+self.muscleLengthScale

    def kneeJointMuscleMomentArm(self, kneeAngle, hipAngle):
        self.kneeAngle = kneeAngle
        self.hipAngle = hipAngle
        self.muscleMomentArm = self.muscleMomentArmParam[0] * self.kneeAngle + self.muscleMomentArmParam[
            1] * self.kneeAngle ** 2 + self.muscleMomentArmParam[2] * self.kneeAngle ** 3 + self.muscleMomentArmParam[
                              3] * self.hipAngle
        return self.muscleMomentArm


class ActiveForceLengthRelationship(object):
    """
        Compute force-length properties
    """

    def __init__(self):
        self.kShapeActive = 0.45

    def thelenActiveMuscleForce(self, muscleLength, muscleOptimaleLength):
        self.muscleOptimalLength = muscleOptimaleLength
        self.muscleLength = muscleLength

        return exp(- (muscleLength / self.muscleOptimalLength - 1) ** 2 / self.kShapeActive)


class TendonForceRelationShip(object):
    """
        Compute tendon force
    """

    def __init__(self):
        self.tendonStrainMaximalForce = 0.033
        self.tendonLinearShapeFactor = 1.712 / self.tendonStrainMaximalForce
        self.tendonStrainLinear =  0.609 * self.tendonStrainMaximalForce
        self.tendonNormalizedMaximalForceLinear = 0.333333
        self.tendonNonlinearShapeFactor = 3

    def thelenTensonForce(self, tendonLength, tendonSlackLength):
        self.tendonLength = tendonLength
        self.tendonSlackLength = tendonSlackLength
        self.relativeTendonLength = ( self.tendonLength - self.tendonSlackLength ) / self.tendonSlackLength
        if self.relativeTendonLength <= 0 :
            self.tendonForce = 0
        elif self.relativeTendonLength > 0 and self.relativeTendonLength <= self.tendonStrainLinear :
            self.tendonForce = self.tendonNormalizedMaximalForceLinear * (exp(self.tendonNonlinearShapeFactor * self.relativeTendonLength / self.tendonStrainLinear) - 1) / ( exp( self.tendonNonlinearShapeFactor ) - 1)
        else :# self.relativeTendonLength > self.tendonStrainLinear :
            self.tendonForce = self.tendonLinearShapeFactor * (self.relativeTendonLength - self.tendonStrainLinear ) + self.tendonNormalizedMaximalForceLinear

        return self.tendonForce + 0.001 * (1 + self.relativeTendonLength)


class PassiveMuscleForceRelationShip(object):

    """
        Compute muscle passive force
    """

    def __init__(self):
        self.passiveExpoShapeFactor = 4 #Kpe
        self.passiveNormalizedMaximalForce = 0.6 #e_O^m

    def thelenPassiveForce(self, muscleLength, muscleOptimaleLength):
        self.muscleLength = muscleLength
        self.muscleOptimaleLength = muscleOptimaleLength
        self.normalizedMuscleLength = self.muscleLength / self.muscleOptimaleLength
        if self.normalizedMuscleLength <= 1 + self.passiveNormalizedMaximalForce :
            passiveMuscleForce = (exp(self.passiveExpoShapeFactor*(self.normalizedMuscleLength-1)/self.passiveNormalizedMaximalForce)) / (exp(self.passiveExpoShapeFactor))
        elif self.normalizedMuscleLength > 1 + self.passiveNormalizedMaximalForce :
            passiveMuscleForce = 1+ self.passiveExpoShapeFactor/self.passiveNormalizedMaximalForce * (self.normalizedMuscleLength-(1+self.passiveNormalizedMaximalForce))
        else :# self.relativeTendonLength > self.tendonStrainLinear :
            passiveMuscleForce = 0

        return passiveMuscleForce
        

class InverseForceVelocityRelatioship(object):

    """
        Compute the inverse force-velocity properties
    """
    def __init__(self):
        self.maxNormalizedMuscleForceLengthening = 1.8
        self.shapeForceVelocityRelationShip = 0.3
        self.dampingFactor = 0.5
        
    def theleninverseForceVelocity(self, activeForce, contractileForce):
        self.velocityDependForce = activeForce/(contractileForce+0.001)

        if self.velocityDependForce < 0:
            self.normalizedMuscleVelocity = (1 + 1 / self.shapeForceVelocityRelationShip)*self.velocityDependForce - 1
        elif self.velocityDependForce >=0 and self.velocityDependForce < 1 :
            self.normalizedMuscleVelocity = (self.velocityDependForce - 1)/(1 + self.velocityDependForce / self.shapeForceVelocityRelationShip )
        elif self.velocityDependForce >= 1 and self.velocityDependForce < 0.95 * self.maxNormalizedMuscleForceLengthening :            
            self.normalizedMuscleVelocity = (self.velocityDependForce - 1 ) * ( self.maxNormalizedMuscleForceLengthening - 1 ) / ((2 + 2/self.shapeForceVelocityRelationShip)*(self.maxNormalizedMuscleForceLengthening - self.velocityDependForce))
        elif self.velocityDependForce >= 0.95 * self.maxNormalizedMuscleForceLengthening and self.velocityDependForce < self.maxNormalizedMuscleForceLengthening :
            self.normalizedMuscleVelocity = (10 * (self.maxNormalizedMuscleForceLengthening - 1))/((1 + 1 / self.shapeForceVelocityRelationShip) * self.maxNormalizedMuscleForceLengthening) * (-18.05 * self.maxNormalizedMuscleForceLengthening + 18 + 20 * self.velocityDependForce * (self.maxNormalizedMuscleForceLengthening - 1)/(self.maxNormalizedMuscleForceLengthening))
        else :
            self.normalizedMuscleVelocity = self.maxNormalizedMuscleForceLengthening
        return self.normalizedMuscleVelocity    

    def schutteInverseForceVelocity(self, activation, contractileForce, forceDependVelocity):
        self.forceDependVelocity = forceDependVelocity
        self.contractileForce = contractileForce
        self.activation = activation
        if self.forceDependVelocity < self.activation*self.contractileForce :
            self.normalizedMuscleVelocity = (1/2)*(self.forceDependVelocity +self.activation*self.contractileForce*self.shapeForceVelocityRelationShip+self.dampingFactor*self.shapeForceVelocityRelationShip-np.sqrt(self.forceDependVelocity*self.forceDependVelocity+2*self.activation*self.contractileForce*self.shapeForceVelocityRelationShip*self.forceDependVelocity-2*self.dampingFactor*self.shapeForceVelocityRelationShip*self.forceDependVelocity+self.activation*self.activation*self.contractileForce*self.contractileForce*self.shapeForceVelocityRelationShip*self.shapeForceVelocityRelationShip+2*self.activation*self.contractileForce*self.shapeForceVelocityRelationShip*self.shapeForceVelocityRelationShip*self.dampingFactor+self.dampingFactor*self.dampingFactor*self.shapeForceVelocityRelationShip*self.shapeForceVelocityRelationShip+4*self.dampingFactor*self.activation*self.contractileForce*self.shapeForceVelocityRelationShip))/self.dampingFactor        
        else:            
            self.normalizedMuscleVelocity = -(0.5e-2*(-100*self.forceDependVelocity+180*self.activation*self.contractileForce+180*self.activation*self.contractileForce*self.shapeForceVelocityRelationShip-100*self.forceDependVelocity*self.shapeForceVelocityRelationShip+13*self.dampingFactor*self.shapeForceVelocityRelationShip-1*np.sqrt(-520*self.dampingFactor*self.activation*self.contractileForce*self.shapeForceVelocityRelationShip-520*self.activation*self.contractileForce*self.shapeForceVelocityRelationShip*self.shapeForceVelocityRelationShip*self.dampingFactor-72000*self.activation*self.contractileForce*self.shapeForceVelocityRelationShip*self.forceDependVelocity+20000*self.forceDependVelocity*self.forceDependVelocity*self.shapeForceVelocityRelationShip+32400*self.activation*self.activation*self.contractileForce*self.contractileForce+10000*self.forceDependVelocity*self.forceDependVelocity*self.shapeForceVelocityRelationShip*self.shapeForceVelocityRelationShip-36000*self.activation*self.contractileForce*self.shapeForceVelocityRelationShip*self.shapeForceVelocityRelationShip*self.forceDependVelocity+32400*self.activation*self.activation*self.contractileForce*self.contractileForce*self.shapeForceVelocityRelationShip*self.shapeForceVelocityRelationShip-36000*self.forceDependVelocity*self.activation*self.contractileForce+64800*self.activation*self.activation*self.contractileForce*self.contractileForce*self.shapeForceVelocityRelationShip+2600*self.forceDependVelocity*self.shapeForceVelocityRelationShip*self.shapeForceVelocityRelationShip*self.dampingFactor+169*self.dampingFactor*self.dampingFactor*self.shapeForceVelocityRelationShip*self.shapeForceVelocityRelationShip+10000*self.forceDependVelocity*self.forceDependVelocity+2600*self.dampingFactor*self.shapeForceVelocityRelationShip*self.forceDependVelocity)))/(self.dampingFactor*(1+self.shapeForceVelocityRelationShip))
        return self.normalizedMuscleVelocity
class ForceVelocityRelatioship(object):
    """
        Compute force-velocity properties
    """
    def __init__(self):
        self.shapeForceVelocityRelationShip = 0.3

    def thelenForceVelocity(self, muscleVelocity, muscleOptimalLength):

        self.muscleOptimalLength = muscleOptimalLength
        self.muscleVelocity = muscleVelocity

        self.normMuscleVelocity = self.muscleVelocity / (10 * self.muscleOptimalLength)
        forceVelocityDependForce = []

#        for i in range(0, len(self.muscleVelocity)):
#            forceVelocityDependForce.append(0)

#        for i in range(0, len(self.muscleVelocity)):
#        if self.normMuscleVelocity[i] <= 0:
#             forceVelocityDependForce[i] = ((self.normMuscleVelocity[i] + 1)) / (-self.normMuscleVelocity[i] / self.shapeForceVelocityRelationShip + 1)
#        else:
#             forceVelocityDependForce[i] = ((1.8 * self.normMuscleVelocity[i] + 0.13 * (self.shapeForceVelocityRelationShip / (self.shapeForceVelocityRelationShip + 1)))) / (self.normMuscleVelocity[i] + 0.13 * (self.shapeForceVelocityRelationShip / (self.shapeForceVelocityRelationShip + 1)))
 
        if self.normMuscleVelocity <= 0:
             forceVelocityDependForce = ((self.normMuscleVelocity + 1)) / (-self.normMuscleVelocity / self.shapeForceVelocityRelationShip + 1)
        else:
             forceVelocityDependForce = ((1.8 * self.normMuscleVelocity + 0.13 * (self.shapeForceVelocityRelationShip / (self.shapeForceVelocityRelationShip + 1)))) / (self.normMuscleVelocity + 0.13 * (self.shapeForceVelocityRelationShip / (self.shapeForceVelocityRelationShip + 1)))
             
        return forceVelocityDependForce


class ComputeVelocity(object):
    """
        Compute velocity
    """

    def __init__(self):
        self.sampleTime = 0.001

    def velovity(self, signal):
        import numpy as np

        self.muscleLength = signal
        velocity = np.diff(self.muscleLength, axis=0) * self.sampleTime
        return np.append(velocity, velocity[-1])


class ComputeActivation(object):
    """
        Compute muscle activation
    """

    def __init__(self):
        self.activationTimeConstant = 0.015
        self.desactivationTimeCOnstatnt = 0.06
        self.maxFiltredEMG = 100
        self.oldActivation = 0


    def yamagushiComputeActivation(self, filteredEMG, maxFiltredEMG):
        self.maxFiltredEMG = maxFiltredEMG
        self.filteredEMG = filteredEMG

        beta = self.activationTimeConstant / self.desactivationTimeCOnstatnt
        self.activationDerivative = []
        for i in range(0, len(self.filteredEMG)):
            self.activationDerivative.append(0)
        self.activation = []
        for i in range(0, len(self.filteredEMG)):
            self.activation.append(0)

        for i in range(0, len(self.filteredEMG)):
            self.activationDerivative[i] = 1 / self.activationTimeConstant * (
                self.filteredEMG[i] / self.maxFiltredEMG - (
                    beta + (1 - beta) * self.filteredEMG[i] / self.maxFiltredEMG) * self.oldActivation)
            self.activation[i] = self.oldActivation + 0.001 * self.activationDerivative[i]
            self.oldActivation = self.activation[i]

        return self.activation


class MuscleActuator():
    """
        Compute muscle activation
    """
    def __init__(self, muscleName, anatomyModel=None, tendonForceRelationShip = None, passiveMuscleForceRelationShip = None, inverseForceVelocityRelatioship = None,  forceLengthModel=None, forceVelocityModel=None):
        self.muscleName = muscleName
        self.muscleTendonLength = 1
        self.tendonSlackLength = 1
        self.computeVelocity = ComputeVelocity()
        self.torqueOffset = 0

        if anatomyModel is None:
            self.anatomyModel = AnatomicalMuscleModel(muscleName)
        else:
            self.anatomyModel = anatomyModel

        if forceLengthModel is None:
            self.forceLengthModel = ActiveForceLengthRelationship()
        else:
            self.forceLengthModel = forceLengthModel

        if forceVelocityModel is None:
            self.forceVelocityModel = ForceVelocityRelatioship()
        else:
            self.forceVelocityModel = forceVelocityModel
            
        if tendonForceRelationShip is None:
            self.tendonForceRelationShip = TendonForceRelationShip()
        else:
            self.tendonForceRelationShip = tendonForceRelationShip

        if inverseForceVelocityRelatioship is None:
            self.inverseForceVelocityRelatioship = InverseForceVelocityRelatioship()
        else:
            self.inverseForceVelocityRelatioship = inverseForceVelocityRelatioship
            
        if passiveMuscleForceRelationShip is None:
            self.passiveMuscleForceRelationShip = PassiveMuscleForceRelationShip()
        else:
            self.passiveMuscleForceRelationShip = passiveMuscleForceRelationShip
        
        self.filteringData = FilteringData()
        
    def staticMuscleModel(self, muscleActivation, kneeAngle, hipAngle):
        #self.muscleName=muscleName
        self.kneeAngle = kneeAngle
        self.hipAngle = hipAngle

        self.muscleLength = self.anatomyModel.kneeJointMuscleLength(self.kneeAngle,self.hipAngle) - self.anatomyModel.tendonSlackLength
        self.muscleMomentArm = self.anatomyModel.kneeJointMuscleMomentArm(self.kneeAngle, self.hipAngle)
        self.muscleVelocity = self.computeVelocity.velovity(self.muscleLength)

        self.velocityForceDependecy = []
        for i in range(len(self.muscleLength)):
            self.velocityForceDependecy.append(1)

        self.velocityForceDependecy = self.forceVelocityModel.thelenForceVelocity(self.muscleVelocity, self.anatomyModel.muscleOptimalLength)

        self.muscleLengthForceRelationShip = self.forceLengthModel.thelenActiveMuscleForce(self.muscleLength, self.anatomyModel.muscleOptimalLength)

        self.muscleActiveForce = self.muscleLengthForceRelationShip * muscleActivation

        self.muscleForce = self.anatomyModel.muscleMaximalIsometricForce * self.muscleActiveForce * self.velocityForceDependecy

        self.muscleTorque = self.muscleForce * self.muscleMomentArm

        return self.muscleTorque

    def theleDynamicMuscleModel(self, muscleActivation, kneeAngle, hipAngle):
        #self.muscleName=muscleName
        self.kneeAngle = kneeAngle
        self.hipAngle = hipAngle
        self.muscleActivation = muscleActivation

        self.muscleTendonLength = self.anatomyModel.kneeJointMuscleLength(self.kneeAngle,self.hipAngle)
        self.muscleTendonMomentArm = self.anatomyModel.kneeJointMuscleMomentArm(self.kneeAngle, self.hipAngle)
        self.tendonSlackLength = self.anatomyModel.tendonSlackLength
        self.muscleOptimalLength = self.anatomyModel.muscleOptimalLength

        self.muscleLength = np.zeros((np.size(self.kneeAngle)))
        self.muscleLength[0] = self.muscleTendonLength[0] - self.tendonSlackLength
#        self.muscleLength[1] = self.muscleTendonLength[0] - self.tendonSlackLength

        self.tendonLength = np.zeros((np.size(self.kneeAngle)))
        self.tendonLength[0] = self.tendonSlackLength
#        self.tendonLength[1] = 2*self.tendonSlackLength
        
        self.muscleActiveForce = np.zeros((np.size(self.kneeAngle)))
        self.tendonForce = np.zeros((np.size(self.kneeAngle)))
        self.musclePassiveForce = np.zeros((np.size(self.kneeAngle)))
        self.muscleActiveForce = np.zeros((np.size(self.kneeAngle)))
        self.muscleContractileForce = np.zeros((np.size(self.kneeAngle)))
        self.velocityDependentForce = np.zeros((np.size(self.kneeAngle)))
        self.muscleVelocity = np.zeros((np.size(self.kneeAngle)))
        self.muscleTorque = np.zeros((np.size(self.kneeAngle)))
        self.tendonLengthTmp = 0
        self.muscleContractilForce = np.zeros((np.size(self.kneeAngle)))
        self.velocityDependentForceTmp = np.zeros((np.size(self.kneeAngle)))
#        self.torqueOffset = 0
        self.muscleVelocityTmp = np.zeros((np.size(self.kneeAngle)))
        
        
        for i in range(1, len(self.kneeAngle)):
            
            # Compute Tendon Force
                            
            self.tendonForce[i] = max(0, self.tendonForceRelationShip.thelenTensonForce(self.tendonLength[i-1], self.tendonSlackLength))
            
            # compute muscle Passive Force            
            self.musclePassiveForce[i] = 0# max(0, self.passiveMuscleForceRelationShip.thelenPassiveForce(self.muscleLength[i-1], self.muscleOptimalLength))

            # compute fiber active force            
            self.muscleActiveForce[i] = max(0, self.muscleActivation[i] * self.forceLengthModel.thelenActiveMuscleForce(self.muscleLength[i-1], self.muscleOptimalLength))# * self.forceVelocityModel.thelenForceVelocity(self.muscleVelocity[i-1], self.anatomyModel.muscleOptimalLength)

            self.muscleContractilForce[i] = self.muscleActiveForce[i] * self.forceVelocityModel.thelenForceVelocity(self.muscleVelocity[i-1], self.anatomyModel.muscleOptimalLength) + self.musclePassiveForce[i]
            
            if self.muscleActiveForce[i] > 0.01:
                self.velocityDependentForceTmp[i] = (self.tendonForce[i] - self.musclePassiveForce[i])  / (self.muscleActiveForce[i])
            else :
                self.velocityDependentForceTmp[i] = 0
            
            self.velocityDependentForce[i] = max(self.velocityDependentForceTmp[i], 0)
            self.velocityDependentForce[i] = min(self.velocityDependentForceTmp[i], 1.8)            
            
#            self.muscleVelocityTmp = self.inverseForceVelocityRelatioship.schutteInverseForceVelocity(self.muscleActivation[i], max(0,self.forceLengthModel.thelenActiveMuscleForce(self.muscleLength[i-1], self.muscleOptimalLength)), self.velocityDependentForce[i])
 
            self.muscleVelocityTmp[i] = self.inverseForceVelocityRelatioship.theleninverseForceVelocity(self.velocityDependentForce[i], self.muscleActiveForce[i])

#            self.muscleVelocity[i] = np.sum(self.muscleVelocityTmp[max(0,i-200): i])/(i-max(0,i-200))

            self.muscleVelocity[i] = min(self.muscleVelocityTmp[i], 1)
            
            self.muscleVelocity[i] = max(self.muscleVelocityTmp[i], -1)                        
            
#            print(self.muscleVelocity[i])
            
            self.muscleLength[i] = self.muscleLength[i-1] + 0.001 * (self.muscleVelocity[i] * 10 * self.muscleOptimalLength)
            
#            self.tendonLength[i] = self.muscleTendonLength[i]-self.muscleLength[i]            

            if self.muscleLength[i] >= 2 * self.muscleOptimalLength :
                self.muscleLength[i] = 2 * self.muscleOptimalLength
            elif self.muscleLength[i] <= 0 * self.muscleOptimalLength :
                self.muscleLength[i] = 0 * self.muscleOptimalLength
#            elif self.muscleLength[i]+self.tendonLength[i] >= self.muscleTendonLength[i]:
#                self.muscleLength[i]=self.muscleTendonLength[i]-self.tendonLength[i]
            else:
                self.muscleLength[i] = self.muscleLength[i]

            self.tendonLength[i] = self.muscleTendonLength[i]-self.muscleLength[i]

            self.muscleTorque[i] =  max(0,self.anatomyModel.muscleMaximalIsometricForce * (self.tendonForce[i] ) * self.muscleTendonMomentArm[i] - self.torqueOffset)
                        
#        self.result = self.filteringData.butterDataFiltering(self.muscleTorque,3.0)    
        return self.muscleTorque #self.result #anatomyModel.muscleMaximalIsometricForce * self.muscleActiveForce


class InverseDynamicsProcessing:
    def __init__(self):
        self.inertia = 0.7
        self.damping = 1
        self.stiffness = 4
        self.gravitationalTorque = 18

    def inverseDynamics(self, kneeAngle, kneeVeocity, kneeAcceleration, kneeAngleRest):
        self.kneeAngle = kneeAngle
        self.kneeVelocity = kneeVeocity
        self.kneeAcceleration = kneeAcceleration
        self.kneeAngleRest = kneeAngleRest
        self.inverseTorque = []

        for i in range(0, len(kneeAngle)):
            self.inverseTorque.append(0)
        for i in range(0, len(self.kneeAngle)):
            self.inverseTorque[i] = kneeAcceleration[i] * self.inertia + kneeVeocity[
                i] * self.damping + self.stiffness * (
                                        self.kneeAngle[i] - self.kneeAngleRest[i]) + self.gravitationalTorque * sin(
                self.kneeAngle[i] - self.kneeAngleRest[i])
        return self.inverseTorque


class FilteringData:
    def __init__(self):
        self.butterFilterOrder = 4
        self.mediaFilterOrder = 200
        self.filterCutOffFrequency = 1.5

    def butterDataFiltering(self, signalToFIlter, filterCutOffFrequency):
        self.filterCutOffFrequency = filterCutOffFrequency
        self.signalToFIlter = signalToFIlter
        b, a = signal.butter(self.butterFilterOrder, self.filterCutOffFrequency * 2 / 1000, 'low', analog=False)
        print(b,a)
        return signal.lfilter(b, a, self.signalToFIlter, axis=-1, zi=None)

    def mediaDataFiltering(self, signalToFIlter, mediaFilterOrder):
        self.mediaFilterOrder = mediaFilterOrder
        self.signalToFIlter = signalToFIlter
        filterWindows = np.ones(self.mediaFilterOrder) * self.mediaFilterOrder
        return signal.lfilter(filterWindows, 1, self.signalToFIlter, axis=-1, zi=None)


class PassiveForceLengthRelatioship():
    pass

