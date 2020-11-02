"""utils.py"""

import matplotlib.pyplot as plt
import os


def plot_results(output_dir, file_name, z_real, z_net, t_vec, sys_name):
    plt.clf()
    t_vec = t_vec[0,:]

    if sys_name == 'double_pendulum':

        fig, axes = plt.subplots(3,2, figsize=(15, 15))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        fig.suptitle(file_name)
   
        l1, = ax1.plot(t_vec, z_net[0,:],'b')
        l2, = ax1.plot(t_vec, z_net[1,:],'r')
        l3, = ax1.plot(t_vec, z_real[0,:],'k--')          
        l4, = ax1.plot(t_vec, z_real[1,:],'k--')
        ax1.legend((l1, l2, l4), ('Net (X)', 'Net (Y)', 'GT'))
        ax1.set_ylabel('$q_1$ [m]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()

        l1, = ax3.plot(t_vec, z_net[2,:],'b')
        l2, = ax3.plot(t_vec, z_net[3,:],'r')
        l3, = ax3.plot(t_vec, z_real[2,:],'k--')          
        l4, = ax3.plot(t_vec, z_real[3,:],'k--')
        ax3.legend((l1, l2, l4), ('Net (X)', 'Net (Y)', 'GT'))
        ax3.set_ylabel('$q_2$ [m]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()

        l1, = ax2.plot(t_vec, z_net[4,:],'b')
        l2, = ax2.plot(t_vec, z_net[5,:],'r')
        l3, = ax2.plot(t_vec, z_real[4,:],'k--')          
        l4, = ax2.plot(t_vec, z_real[5,:],'k--')
        ax3.legend((l1, l2, l4), ('Net (X)', 'Net (Y)', 'GT'))
        ax2.set_ylabel('$p_1$ [kg·m/s]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()

        l1, = ax4.plot(t_vec, z_net[6,:],'b')
        l2, = ax4.plot(t_vec, z_net[7,:],'r')
        l3, = ax4.plot(t_vec, z_real[6,:],'k--')          
        l4, = ax4.plot(t_vec, z_real[7,:],'k--')
        ax4.legend((l1, l2, l4), ('Net (X)', 'Net (Y)', 'GT'))
        ax4.set_ylabel('$p_2$ [kg·m/s]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()

        l1, = ax5.plot(t_vec, z_net[8,:],'b')
        l2, = ax5.plot(t_vec, z_real[8,:],'k--')
        ax5.legend((l1, l2), ('Net','GT'))
        ax5.set_ylabel('$s_1$ [J/K]')
        ax5.set_xlabel('$t$ [s]')
        ax5.grid()

        l1, = ax6.plot(t_vec, z_net[9,:],'r')
        l2, = ax6.plot(t_vec, z_real[9,:],'k--')
        ax6.legend((l1, l2), ('Net','GT'))
        ax6.set_ylabel('$s_2$ [J/K]')
        ax6.set_xlabel('$t$ [s]')
        ax6.grid()

        save_dir = os.path.join(output_dir, '[Double Pendulum] ' + file_name)

    elif sys_name == 'viscolastic':

        fig, axes = plt.subplots(2,2, figsize=(15, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()
        fig.suptitle(file_name)

        l1, = ax1.plot(t_vec, z_net[0,:],'b')
        l2, = ax1.plot(t_vec, z_real[0,:],'k--')
        ax1.legend((l1, l2), ('Net','GT'))
        ax1.set_ylabel('$q$ [-]')
        ax1.set_xlabel('$t$ [-]')
        ax1.grid()

        l1, = ax2.plot(t_vec, z_net[2,:],'b')
        l2, = ax2.plot(t_vec, z_real[2,:],'k--')
        ax2.legend((l1, l2), ('Net','GT'))
        ax2.set_ylabel('$v$ [-]')
        ax2.set_xlabel('$t$ [-]')
        ax2.grid()

        l1, = ax3.plot(t_vec, z_net[3,:],'b')
        l2, = ax3.plot(t_vec, z_real[3,:],'k--')
        ax3.legend((l1, l2), ('Net','GT'))
        ax3.set_ylabel('$e$ [-]')
        ax3.set_xlabel('$t$ [-]')
        ax3.grid()

        l1, = ax4.plot(t_vec, z_net[4,:],'b')
        l2, = ax4.plot(t_vec, z_real[4,:],'k--')
        ax4.legend((l1, l2), ('Net','GT'))
        ax4.set_ylabel('$\\tau$ [-]')
        ax4.set_xlabel('$t$ [-]')
        ax4.grid()

        save_dir = os.path.join(output_dir, '[Viscolastic] ' + file_name)

    plt.savefig(save_dir)
    plt.clf()
