import sim as vrep # access all the VREP elements
import sys 
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # start a connection

if clientID!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")

err_code,camera = vrep.simxGetObjectHandle(clientID,"/NAO/vision[0]", vrep.simx_opmode_blocking)

err_code,resolution,image = vrep.simxGetVisionSensorImage(clientID, camera,0,vrep.simx_opmode_streaming)
print(err_code)