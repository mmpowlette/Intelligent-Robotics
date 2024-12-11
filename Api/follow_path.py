import numpy as np
import sim



client = sim.simxStart('127.0.0.1',19999,True,True,5000,5)

def sysCall_init():
    client.objectToFollowPath = sim.getObject('/Nao')
    client.path = sim.getObject('/Path')
    pathData = sim.unpackDoubleTable(sim.getBufferProperty(client.path, 'customData.PATH'))

    m = np.array(pathData).reshape(len(pathData) // 7, 7)
    client.pathPositions = m[:, :3].flatten().tolist()
    client.pathQuaternions = m[:, 3:].flatten().tolist()

    client.pathLengths, client.totalLength = sim.getPathLengths(client.pathPositions, 3)
    client.velocity = 0.04 # m/s
    client.posAlongPath = 0
    client.previousSimulationTime = 0
    sim.setStepping(True)

def sysCall_thread():
    while not sim.getSimulationStopping():
        t = sim.getSimulationTime()
        client.posAlongPath += client.velocity * (t - client.previousSimulationTime)
        client.posAlongPath %= client.totalLength
        pos = sim.getPathInterpolatedConfig(client.pathPositions, client.pathLengths, client.posAlongPath)
        quat = sim.getPathInterpolatedConfig(client.pathQuaternions, client.pathLengths,
               client.posAlongPath, None, [2, 2, 2, 2])
        sim.setObjectPosition(client.objectToFollowPath, pos, client.path)
        sim.setObjectQuaternion(client.objectToFollowPath, quat, client.path)
        client.previousSimulationTime = t
        sim.step()
