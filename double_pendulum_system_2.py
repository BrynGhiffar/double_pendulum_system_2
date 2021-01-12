from math import sin, cos, pi
from ast import literal_eval
import time
import matplotlib.pyplot as plt
import sys
from IPython.display import clear_output
import numpy as np
import os
from scipy.integrate import solve_ivp


class double_pendulum_system:


    def __init__(self, m1, l1, th1, th1_dot, th1_dot2,      
                    m2, l2,th2, th2_dot, th2_dot2,      
                    g, inc, time_start, time_end):
        # Creating the object will set it's initial conditions
        self.m1 = m1
        self.l1 = l1
        self.th1 = np.array([th1])
        self.th1_dot = np.array([th1_dot])
        self.th1_dot2 = np.array([th1_dot2])

        self.x1 = np.array([])
        self.y1 = np.array([])

        self.m2 = m2
        self.l2 = l2
        self.th2 = np.array([th2])
        self.th2_dot = np.array([th2_dot])
        self.th2_dot2 = np.array([th2_dot2])
        self.x2 = np.array([])
        self.y2 = np.array([])

        self.g = g
        self.inc = inc
        self.time_start = time_start
        self.time_end = time_end
        self.time_stamp_amount = int((self.time_end - self.time_start) / self.inc)
    def update_progress(self, progress, section):
        bar_length = 50
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1
        block = int(round(bar_length * progress))
        clear_output(wait = True)
        text = section + ": [{0} {1:.1f}%] ".format("#" * block + "-" * (bar_length - block), progress * 100)
        print(text)

    def pend_diff_1(self): # Differential for first pendulum
            top = np.sum([
                -self.g * (2 * self.m1 + self.m2) * np.sin(self.th1[-1]),
                - self.m2 * self.g * np.sin(self.th1[-1] - 2 * self.th2[-1]),
                -2 * np.sin(self.th1[-1] - self.th2[-1]) * self.m2 * (self.th2_dot[-1] ** 2 * self.l2),
                -2 * np.sin(self.th1[-1] - self.th2[-1]) * self.m2 * (self.th1_dot[-1] ** 2 * self.l1 * np.cos(self.th1[-1] - self.th2[-1]))
            ])
            bottom = np.sum([
                self.l1 * 2 * self.m1,
                self.l1 * self.m2 ,
                - self.l1 * self.m2 * np.cos(2 * self.th1[-1] - 2 * self.th2[-1])
            ])
            return top / bottom
        

    def pend_diff_2(self): # Differential for second pendulum
        top = np.sum([
            2 * np.sin(self.th1[-1] - self.th2[-1]) * self.th1_dot[-1] ** 2 * self.l1 * (self.m1 + self.m2),
            2 * np.sin(self.th1[-1] - self.th2[-1]) * self.g * (self.m1 + self.m2) * np.cos(self.th1[-1]),
            2 * np.sin(self.th1[-1] - self.th2[-1]) * self.th2_dot[-1] ** 2 * self.l2 * self.m2 * np.cos(self.th1[-1] - self.th2[-1])
        ])
        bottom = np.sum([
            self.l2 * (2 * self.m1  + self.m2 - self.m2 * np.cos(2 * self.th1[-1] - 2 * self.th2[-1]))
        ])
        return top / bottom
    

    def generate_soln(self):
        for i in range(1, self.time_stamp_amount):
            
            self.update_progress(i / self.time_stamp_amount, "Solving equations")

            self.th1 = np.append(self.th1, self.th1_dot[-1] * (self.inc) + self.th1[-1])
            self.th2 = np.append(self.th2, self.th2_dot[-1] * (self.inc) + self.th2[-1])
            self.th1_dot = np.append(self.th1_dot, self.th1_dot2[-1] * (self.inc) + self.th1_dot[-1])
            self.th2_dot = np.append(self.th2_dot, self.th2_dot2[-1] * (self.inc) + self.th2_dot[-1])
            self.th1_dot2 = np.append(self.th1_dot2, self.pend_diff_1())
            self.th2_dot2 = np.append(self.th2_dot2, self.pend_diff_2())
        


    def convert_to_cartesian(self):
        self.update_progress(0 / 4, "Converting to cartesian")
        self.x1 = self.l1 * np.sin(self.th1)
        self.update_progress(1 / 4, "Converting to cartesian")
        self.y1 = - self.l1 * np.cos(self.th1)
        self.update_progress(2 / 4, "Converting to cartesian")
        self.x2 = self.x1 + self.l2 * np.sin(self.th2)
        self.update_progress(3 / 4, "Converting to cartesian")
        print(len(self.th1))
        print(len(self.th2))
        self.y2 =  self.y1 - self.l2 * np.cos(self.th2)
        self.update_progress(4 / 4, "Converting to cartesian")



    
    

    def solve(self):
        self.th1_dot2 = np.array([self.pend_diff_1()])
        self.th2_dot2 = np.array([self.pend_diff_2()])
        print("Generating solution")
        self.generate_soln()
        print("Finished solution")
        print("Converting to cartesian")
        self.convert_to_cartesian()
        print("Finished converting to cartesian")


    def write_all_soln(self, file_name_ext):
        prev_dir = os.path.abspath('.')
        os.mkdir(file_name_ext)
        os.chdir(prev_dir + '\\' + file_name_ext)
        np.save('dp_sys_ang1_{}.npy'.format(file_name_ext), self.th1)
        np.save('dp_sys_ang2_{}.npy'.format(file_name_ext), self.th2)
        np.save('dp_sys_ang1_vel_{}.npy'.format(file_name_ext), self.th1_dot)
        np.save('dp_sys_ang2_vel_{}.npy'.format(file_name_ext), self.th1_dot)
        np.save('dp_sys_ang1_acc_{}.npy'.format(file_name_ext), self.th1_dot2)
        np.save('dp_sys_ang2_acc_{}.npy'.format(file_name_ext), self.th1_dot2)
        np.save('dp_sys_x1_{}.npy'.format(file_name_ext), self.x1)
        np.save('dp_sys_y1_{}.npy'.format(file_name_ext), self.y1)
        np.save('dp_sys_x2_{}.npy'.format(file_name_ext), self.x2)
        np.save('dp_sys_y2_{}.npy'.format(file_name_ext), self.y2)
        os.chdir(prev_dir)


def plot_soln(file_name_ext):
    print("Plotting trajectory")
    prev_path = os.path.abspath('.')
    os.chdir(prev_path + '\\' + file_name_ext)
    x1 = np.load('dp_sys_x1_{}.npy'.format(file_name_ext))
    y1 = np.load('dp_sys_y1_{}.npy'.format(file_name_ext))
    x2 = np.load('dp_sys_x2_{}.npy'.format(file_name_ext))
    y2 = np.load('dp_sys_y2_{}.npy'.format(file_name_ext))
    plt.plot(x1, y1, label = "Pend 1")
    plt.plot(x2, y2, label = "Pend 2")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend(loc = 'upper left')
    plt.title(file_name_ext)
    plt.show()
    print("Done plotting trajectory")
    os.chdir(prev_path)

