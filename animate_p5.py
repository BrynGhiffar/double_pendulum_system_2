from double_pendulum_system_2 import *
import os
# print(os.getcwd())
file_ext = '1'
os.chdir('C:\\Users\\bryng\\OneDrive - Monash University\\PythonPrograms\\pendulum_sim\\double_pendulum_system_2\\{}'.format(file_ext))
# print(os.getcwd())
x1 = np.load('dp_sys_x1_{}.npy'.format(file_ext))
y1 = np.load('dp_sys_y1_{}.npy'.format(file_ext))
x2 = np.load('dp_sys_x2_{}.npy'.format(file_ext))
y2 = np.load('dp_sys_y2_{}.npy'.format(file_ext))
# print(len(x1))

from p5 import *

i = 0 
def setup():
    
    size(500, 500)
    background(255)
    stroke(0)

def draw():
    global i
    print(frame_rate)
    background(255)
    stroke(0)
    circle(((100 * x1[i] + 250), ( - 100 * y1[i] + 100)), 20)
    circle(((100 * x2[i] + 250), ( - 100 * y2[i] + 100)), 20)
    i += round((40 * 10 ** -3) ** -1) # The value next to frame rate is the time increment in the calculation
    

if __name__ == '__main__':
    run()