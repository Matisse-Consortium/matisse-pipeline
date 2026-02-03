#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#!/usr/bin/python
  $Id: mat_show_opd.py 2018-03-14 author: jvarga

  This file is part of the Matisse pipeline GUI series
  Copyright (C) 2017- Observatoire de la Côte d'Azur

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

  $Author: jvarga $
  $Date: 2018-03-14 $
  $Revision: 1 $
  $Name:  $
  first line in case of Windows: #!/cygdrive/d/jvarga/Programok/Anaconda/python
"""

# This script can be run from the shell command line or from the python command interface
#usage:
#   with no arguments given: it launches a graphical open file dialog window to select the input file
#   with 1 arguments: argument is the input filename
#   with 2 arguments: 1st argument is the input filename, 2nd argument is the output folder

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
import wx
import sys

# basic parameters
output_plot1_name = "_OPDplot"
n_baselines = 6
colors = ['red','orange','green','blue','purple','black']

#open a file browse dialog to select which file to open
def get_path():
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', style=style) #wildcard=wildcard
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

#main function
def mat_show_opd(input_path="",output_path=""):
    if input_path == "":
          input_path = get_path()
    # open input fits file
    abs_path = os.path.abspath(input_path)
    input_path_dir = os.path.dirname(abs_path)
    input_filename = os.path.basename(abs_path)
    hdu_list = fits.open(abs_path, mode='update')

    #extract data from file
    time = np.array(hdu_list['OPD'].data['TIME'])
    time=time-time[0]
    opd = hdu_list['OPD'].data['OPD']
    opd = np.array(opd)
    print(opd)
    N_tot = np.size(opd)
    print(N_tot)
    print(N_tot//n_baselines)
    opd = np.reshape(opd,[N_tot//n_baselines,n_baselines])

    #mjd = np.reshape(mjd,[N_tot/n_baselines,n_baselines])
    #make OPD plot
    fig1, (ax1) = plt.subplots(1, 1, sharey=False, figsize=(10, 8))
    for j in range(n_baselines):
        ax1.plot(time[:], opd[:,j], '-', color=colors[j], lw=1.5, alpha=0.6, label=('%d' % (j + 1)))
    ax1.set_xlabel(r'$\mathrm{time}$ (s)')
    ax1.set_ylabel(r"${\mathrm{OPD}}$ ($\mu$m)")
    ax1.set_title(r"$\mathrm{"+ input_filename.split('.')[0].replace('_','\_') +"}$") #replace(r"\",r"\")
    leg = ax1.legend(loc='best') #loc='upper left')
    leg.get_frame().set_alpha(0.5)
    #ax1.set_ylim([0, 3.0*np.nanmedian(SNR[SNR>0])])
    #ax1.set_xlim([np.nanmin(Q)-0.1,1.2])
    plt.tight_layout()

    if output_path == "":
        output_dir = input_path_dir
    else:
        output_dir = output_path
    outputfig = output_dir + '/' + input_filename.split('.')[0] + output_plot1_name
    fig1.savefig(outputfig + '.png', dpi=300)
    fig1.savefig(outputfig + '.eps', format='eps', dpi=300)
    plt.close(fig1)
    print('READY')

if __name__ == '__main__':
    list_arg = sys.argv
    #print list_arg
    nargs = len(list_arg)
    if nargs == 1:
        mat_show_opd()
    elif nargs == 2:
        in_path = list_arg[1]
        mat_show_opd(in_path)
    elif nargs == 3:
        in_path = list_arg[1]
        out_path = list_arg[2]
        mat_show_opd(in_path,out_path)
    else:
        print("Wrong number of arguments.")
