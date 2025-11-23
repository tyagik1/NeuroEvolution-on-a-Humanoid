import pybullet as p  
import time
import pybullet_data
import numpy as np
import neuralNetwork as nn
import perceptron as pp
import eas

def tanh(x):
    x = np.clip(x, -50, 50)
    return np.tanh(x)

def makeArray(value, size):
    array = []
    for i in range(size):
        array.append(value)
    return array

def fitnessFunction(gene):

    p.connect(p.DIRECT)
    p.resetSimulation()

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.86)

    robotId = p.loadURDF("robot.urdf")
    planeId = p.loadURDF("plane.urdf")

    p.changeDynamics(planeId, -1, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.01)

    for i in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, i, lateralFriction=2.0, rollingFriction=0.01, spinningFriction=0.01)
    
    # For CTRNN
    brain = nn.NeuralNetwork(441, 512, 14)
    brain.setParams(gene, barriers)
    brain.initializeState(np.zeros(512))
    for t in range(100):
        brain.step(0.01, makeArray(0.0, 441))
        p.stepSimulation()

    # For FNN
    # brain = pp.Perceptron(441, [512, 256, 128, 64, 32], 14, tanh)
    # brain.setParams(gene)

    duration = 250
    for t in range(duration):
        input = []

        for i in range(15):
            linkState = p.getLinkState(robotId, i, computeLinkVelocity=True)
            if linkState is None:
                input.extend([0] * 21)
                continue
            for j in range(6):
                for k in range(3 + j % 2):
                    val = linkState[j][k] if linkState[j] is not None and linkState[j][k] is not None else 0
                    input.append(val)
        
        for i in range(14):
            jointState = p.getJointState(robotId, i)
            if jointState is None:
                input.extend([0] * 9)
                continue
            for j in range(4):
                if j == 2:
                    if jointState[j] is not None:
                        for k in range(6):
                            val = jointState[j][k] if jointState[j][k] is not None else 0
                            input.append(val)
                    else:
                        input.extend([0] * 6)
                else:
                    val = jointState[j] if jointState[j] is not None else 0
                    input.append(val)

        # For CTRNN
        brain.step(0.01, np.array(input))
        output = brain.out()

        # For FNN
        # brain.forward(input)
        # output = brain.output

        maxTorque = [150, 120, 80, 40, 120, 80, 40, 60, 40, 20, 60, 40, 20, 15]
        maxAngle = [[-0.19*np.pi, 0.10*np.pi], [-0.19*np.pi, 0.10*np.pi], [-0.32*np.pi, 0.13*np.pi], [-0.32*np.pi, 0.13*np.pi], [0, 0.51*np.pi], [0, 0.51*np.pi], [-0.13*np.pi, 0.13*np.pi], [-0.13*np.pi, 0.13*np.pi], [-0.32*np.pi, 0.32*np.pi], [-0.32*np.pi, 0.32*np.pi], [0, 0.64*np.pi], [0, 0.64*np.pi], [-0.16*np.pi, 0.16*np.pi], [-0.22*np.pi, 0.22*np.pi]]

        for i in range(14):
            info = p.getJointInfo(robotId, i)
            lower = info[8]
            upper = info[9]
            angle = (output[i] + 1) / 2 * (upper - lower) + lower
            angle = np.clip(angle, maxAngle[i][0], maxAngle[i][1])
            p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition=angle, force=maxTorque[i], positionGain=0.8, velocityGain=0.3)

        p.stepSimulation()

        base_pos, base_orn = p.getBasePositionAndOrientation(robotId)
        x, y, z = base_pos
        if z < 1.0:
            break
        fitness = x - abs(y) * 0.1

    p.disconnect()

    return fitness

def testRobot():
    gene = np.load(f"gene.npy")

    p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.86)

    robotId = p.loadURDF("robot.urdf")
    p.loadURDF("plane.urdf")

    # For CTRNN
    brain = nn.NeuralNetwork(441, 512, 14)
    brain.setParams(gene, barriers)
    brain.initializeState(np.zeros(512))
    for t in range(100):
        brain.step(0.01, makeArray(0.0, 441))
        p.stepSimulation()

    # For FNN
    # brain = pp.Perceptron(441, [512, 256, 128, 64, 32], 14, tanh)
    # brain.setParams(gene)

    duration = 1000
    for m in range(duration):
        t = m/60

        input = []

        for i in range(15):
            linkState = p.getLinkState(robotId, i, computeLinkVelocity=True)
            if linkState is None:
                input.extend([0] * 21)
                continue
            for j in range(6):
                for k in range(3 + j % 2):
                    val = linkState[j][k] if linkState[j] is not None and linkState[j][k] is not None else 0
                    input.append(val)
        
        for i in range(14):
            jointState = p.getJointState(robotId, i)
            if jointState is None:
                input.extend([0] * 9)
                continue
            for j in range(4):
                if j == 2:
                    if jointState[j] is not None:
                        for k in range(6):
                            val = jointState[j][k] if jointState[j][k] is not None else 0
                            input.append(val)
                    else:
                        input.extend([0] * 6)
                else:
                    val = jointState[j] if jointState[j] is not None else 0
                    input.append(val)

        # For CTRNN
        brain.step(0.01, np.array(input))
        output = brain.out()

        # For FNN
        # brain.forward(input)
        # output = brain.output

        maxTorque = [150, 120, 80, 40, 120, 80, 40, 60, 40, 20, 60, 40, 20, 15]
        maxAngle = [[-0.19*np.pi, 0.10*np.pi], [-0.19*np.pi, 0.10*np.pi], [-0.32*np.pi, 0.13*np.pi], [-0.32*np.pi, 0.13*np.pi], [0, 0.51*np.pi], [0, 0.51*np.pi], [-0.13*np.pi, 0.13*np.pi], [-0.13*np.pi, 0.13*np.pi], [-0.32*np.pi, 0.32*np.pi], [-0.32*np.pi, 0.32*np.pi], [0, 0.64*np.pi], [0, 0.64*np.pi], [-0.16*np.pi, 0.16*np.pi], [-0.22*np.pi, 0.22*np.pi]]

        for i in range(14):
            info = p.getJointInfo(robotId, i)
            lower = info[8]
            upper = info[9]
            angle = (output[i] + 1) / 2 * (upper - lower) + lower
            angle = np.clip(angle, maxAngle[i][0], maxAngle[i][1])
            p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition=angle, force=maxTorque[i], positionGain=0.8, velocityGain=0.3)

        p.stepSimulation()
        time.sleep(1 / 60)
        base_pos, base_orn = p.getBasePositionAndOrientation(robotId)
        x, y, z = base_pos
        if z < 1.0:
            break
        fitness = x - abs(y) * 0.1

    p.disconnect()

    print(f"Fitnes: {fitness}")

def runRobot():
    geneSize = hidden*(hidden + inputs + outputs + 2) # UOnly for CTRNN, comment for FNN
    mutationProbability = 0.5
    recombinationProbability = 0.1
    generations = 10
    population = 5

    robot = eas.EvolutionaryAlgorithm(fitnessFunction, geneSize, barriers, mutationProbability, recombinationProbability, generations, population)

    robot.run()

    # testRobot() # Uncomment for simple run

inputs = 441
outputs = 14

# For CTRNN
hidden = 512
barriers = []
for i in range(hidden * hidden):
    barriers.append([-5, 5])
for i in range(hidden * inputs):
    barriers.append([-10, 10])
for i in range(hidden * outputs):
    barriers.append([-5, 5])
for i in range(hidden):
    barriers.append([-2, 2])
for i in range(hidden):
    barriers.append([0.1, 3.0])

# For FNN
# hidden = [512, 256, 128, 64, 32]
# geneSize = 0
# for i in range(len(hidden)):
#     if i == 0:
#         geneSize += hidden[i]*(inputs + 1)
#     else:
#         geneSize += hidden[i]*(hidden[i - 1] + 1)
# geneSize += outputs*(hidden[-1] + 1)
# paramB = 2
# barriers = []
# for i in range(geneSize):
#     barriers.append([-paramB, paramB])

# runRobot() # uncomment to run
testRobot() # only use to only run the visualizable tests
