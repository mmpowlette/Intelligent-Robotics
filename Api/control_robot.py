import sim 
import time 
import sys


print("program started ")

sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1',19999, True, True, 5000, 5)

if (clientID!= -1):
    print('Connected')
else:
    sys.exit('Failed')

time.sleep(1)

# error_Code, first_camera =sim.simxGetObjectHandle(clientID, '/NAO/vision[0]', sim.simx_opmode_oneshot_wait)
# error_Code, second_camera =sim.simxGetObjectHandle(clientID, '/NAO/vision[1]', sim.simx_opmode_oneshot_wait)


#err_code,resolution,image = sim.simxGetVisionSensorImage(clientID, second_camera,0,sim.simx_opmode_streaming)    ### why is there no image ???

# print(err_code)
# print(resolution)
# print(image)

error_Code, robot =sim.simxGetObjectHandle(clientID, '/NAO/script1', sim.simx_opmode_oneshot_wait)
print(error_Code)


# error_Code,right =sim.simxGetObjectHandle(clientID, '/PioneerP3DX/rightMotor', sim.simx_opmode_oneshot_wait)
