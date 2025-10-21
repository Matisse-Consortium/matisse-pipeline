#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  This file is part of the Matisse pipeline GUI series
  Copyright (C) 2017- Observatoire de la Côte d'Azur

  Created on Sat Mar 17 06:39:49 2018
  @author: fmillour, jvarga

  Please contact florentin.millour@oca.eu for any question

  This software is a computer program whose purpose is to show oifits
  files from the MATISSE instrument.

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.

  You can use, modify and/ or redistribute the software under the
  terms of the CeCILL license as circulated by CEA, CNRS and INRIA at
  the following URL "http://www.cecill.info". You have a copy of the
  licence in the LICENCE.md file.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.

"""

import sys
import wx
import math
import numpy as np
from   matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from   mat_fileDialog import mat_FileDialog
from   mat_fileDialog import identifyFile
from   astropy.io import fits as fits
import os
import glob
import libRobust as robust
from libShowOifits import *
#from astroquery.simbad import Simbad
from astropy import coordinates
from os.path import expanduser
from matplotlib.ticker import *

home = expanduser("~")


###############################################################################
# Example code for TF and VIS plots.
# name_dir = r"D:\jvarga\Dokumentumok\MATISSE\data\OIFITS/"
# # outputdir = r"D:/jvarga/Dokumentumok/MATISSE/data/OIFITS/"
# list_of_dicts = open_oi_dir(name_dir+"2018-03-15")
# filtered_list_of_dicts = filter_oi_list(list_of_dicts,dates=["2018-03-15"],bands=['L'],spectral_resolutions=['LOW'],DIT_range=[0.045,0.055],targets=[])
# show_vis2_tf2_vs_time(filtered_list_of_dicts, wlenRange=[3.6,4.0], showvis=False, saveplots=False)

# dates=["2018-03-14","2018-03-15","2018-03-16","2018-03-11","2018-03-13"]
# bands=['L','M', 'N']
# sp_res = ['MED','LOW','HIGH']
# wlenRange = [[3.6,4.0],[4.6,4.8],[10.0,11.0]]
# DITs = np.array([0.2,0.02,0.05,0.1])
# dt = 0.001
# DITranges = np.transpose(np.array([DITs-dt,DITs+dt])).tolist()
# i=0
# j=0
# k=0
# l=0
# for i in range(len(dates)):
#     list_of_dicts = open_oi_dir(name_dir + dates[i])
#     for j in range(len(bands)):
#         for k in range(len(sp_res)):
#             for l in range(len(DITranges)):
#                 filtered_list_of_dicts = filter_oi_list(list_of_dicts,dates=[dates[i]],bands=[bands[j]],spectral_resolutions=[sp_res[k]],DIT_range=DITranges[l],targets=[])
#                 #show_oi_vs_time(filtered_list_of_dicts, [3.5, 3.95], key="VIS2", datatype='VIS2') #[3.5, 3.95] [10.2,10.9]
#                 print "Selected",len(filtered_list_of_dicts),"objects"
#                 fname = dates[i] + "_" + bands[j] + "_" + sp_res[k] + "_DIT" + ("%.2f"%(DITs[l])).replace('.','_')
#                 #show_oi_vs_wlen(filtered_list_of_dicts[3],datatype="VIS2")
#                 show_vis2_tf2_vs_time(filtered_list_of_dicts,wlenRange=wlenRange[j],showvis=False,saveplots=True,output_path=outputdir+fname) # wlenRange=[3.5, 3.95]);
#
# raise SystemExit

class oi_data_select_frame(wx.Frame):
    wl_min_def = np.array([1.175,1.5, 2.05,3.0,4.5, 7.8,3.0]) #um, J,H,K,L,M,N,LM
    wl_max_def = np.array([1.325,1.75,2.35,4.2,5.0,12.0,5.0 ]) #um, J,H,K,L,M,N,LM
    DIT       = 0.2
    DIT_range = 0.4
    wl_min = wl_min_def[6]
    wl_max = wl_max_def[6]
    date   = "2018-03-16"
    bands  = np.array(['L','M','LM','N'])
    spectral_resolutions = np.array(['LOW','MED','HIGH'])
    name_file       = ""
    name_dir        = ""
    target_selected = ""

    ###########################################################################

    def __init__(self, *args, **kwds):
        self.dic = {}
        self.list_of_dicts = [{}]
        self.filtered_list_of_dicts = [{}]

        # begin wxGlade: oi_data_select_frame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((420, 559))

        self.statusbar = self.CreateStatusBar(1)
        self.statusbar.SetStatusText('')

        # Menu Bar
        #self.frame_menubar = wx.MenuBar()
        #wxglade_tmp_menu   = wx.Menu()
        #self.frame_menubar.Append(wxglade_tmp_menu, "File")
        #self.SetMenuBar(self.frame_menubar)
        # Menu Bar end
        self.panel = wx.Panel(self, wx.ID_ANY)
        self.btn_open_oifits = wx.Button(self.panel, 15, "Open OIFITS data")
        self.btn_load_oifits = wx.Button(self.panel, 14, "Load data")
        self.tb_path    = wx.TextCtrl(self.panel, wx.ID_ANY, "")
        self.tb_target  = wx.TextCtrl(self.panel, wx.ID_ANY, "")
        self.tb_date    = wx.TextCtrl(self.panel, wx.ID_ANY, self.date)
        self.tb_DIT     = wx.TextCtrl(self.panel, wx.ID_ANY, "%.2f"%(self.DIT))
        self.tb_DIT_range = wx.TextCtrl(self.panel, wx.ID_ANY, "%.3f"%(self.DIT_range))
        #self.cb_disp_LOW  = wx.CheckBox(self.panel, wx.ID_ANY, "LOW")
        #self.cb_disp_MED  = wx.CheckBox(self.panel, wx.ID_ANY, "MED")
        #self.cb_disp_HIGH = wx.CheckBox(self.panel, wx.ID_ANY, "HIGH")
        self.cb_disp_LOW  = wx.RadioButton(self.panel, wx.ID_ANY, "LOW", style=wx.RB_GROUP)
        self.cb_disp_MED  = wx.RadioButton(self.panel, wx.ID_ANY, "MED")
        self.cb_disp_HIGH = wx.RadioButton(self.panel, wx.ID_ANY, "HIGH")

        self.cb_pl_OI_t = wx.RadioButton(self.panel, wx.ID_ANY, "Plot OI vs. time", style=wx.RB_GROUP)
        self.cb_pl_OI_wl = wx.RadioButton(self.panel, wx.ID_ANY, "Plot OI vs. wl.")

        self.cb_b_J  = wx.CheckBox(self.panel, wx.ID_ANY, "J")
        self.cb_b_H  = wx.CheckBox(self.panel, wx.ID_ANY, "H")
        self.cb_b_K  = wx.CheckBox(self.panel, wx.ID_ANY, "K")
        self.cb_b_L  = wx.CheckBox(self.panel, wx.ID_ANY, "L")
        self.cb_b_M  = wx.CheckBox(self.panel, wx.ID_ANY, "M")
        self.cb_b_N  = wx.CheckBox(self.panel, wx.ID_ANY, "N")
        self.cb_b_LM = wx.CheckBox(self.panel, wx.ID_ANY, "LM")

        self.tb_wl_min = wx.TextCtrl(self.panel, wx.ID_ANY, "3.6")
        self.tb_wl_max = wx.TextCtrl(self.panel, wx.ID_ANY, "4.0")
        self.btn_def_wl  = wx.Button(self.panel, 13, "Default wl")

        self.cb_errorbar = wx.CheckBox(self.panel, wx.ID_ANY, "Plot errorbars")

        self.btn_CFXAMP = wx.Button(self.panel, 16, "CFXAMP")
        self.btn_CFX2   = wx.Button(self.panel, 17, "CFX2")
        self.btn_VISAMP = wx.Button(self.panel, 0, "VISAMP")
        self.btn_VIS2   = wx.Button(self.panel, 1, "VIS2")
        self.btn_VISPHI = wx.Button(self.panel, 2, "VISPHI")
        self.btn_VIS    = wx.Button(self.panel, 7, "VIS")
        self.btn_T3AMP  = wx.Button(self.panel, 4, "T3AMP")
        self.btn_TF2    = wx.Button(self.panel, 3, "TF2")
        self.btn_T3PHI  = wx.Button(self.panel, 6, "T3PHI")
        self.btn_TF     = wx.Button(self.panel, 8, "TF")
        self.btn_FLUX   = wx.Button(self.panel, 5, "FLUX")
        self.btn_VIS2_vs_freq = wx.Button(self.panel, 9, "VIS2 vs. spatial frequency")
        self.btn_VIS_vs_freq = wx.Button(self.panel, 10, "VIS vs. spatial frequency")
        self.btn_vis2_tf2_cp_vs_time = wx.Button(self.panel, 11, "VIS2, TF2, T3PHI vs. time")
        self.btn_vis_tf_cp_vs_time = wx.Button(self.panel, 12, "VIS, TF, T3PHI vs. time")

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    ###########################################################################

    def __set_properties(self):
        # begin wxGlade: oi_data_select_frame.__set_properties
        self.SetTitle("OIFITS plotter")
        self.tb_path.SetMinSize((134, 23))
        self.tb_wl_min.SetMinSize((60, 23))
        self.tb_wl_max.SetMinSize((60, 23))
        self.btn_VIS_vs_freq.SetMinSize((153, 26))
        self.btn_vis2_tf2_cp_vs_time.SetMinSize((153, 26))
        self.btn_vis_tf_cp_vs_time.SetMinSize((153, 26))
        # end wxGlade

        self.cb_disp_MED.SetValue(True)
        self.cb_b_LM.SetValue(True)
        self.cb_errorbar.SetValue(True)
        self.cb_pl_OI_wl.SetValue(True)
        self.Bind(wx.EVT_BUTTON, self.OnButtonClicked)

    ###########################################################################

    def __do_layout(self):
        # begin wxGlade: oi_data_select_frame.__do_layout
        sizer_1      = wx.BoxSizer(wx.VERTICAL)
        grid_sizer_1 = wx.FlexGridSizer(1, 2, 0, 0)
        sizer_4      = wx.StaticBoxSizer(wx.StaticBox(self.panel, wx.ID_ANY,
                                                      "Band"), wx.VERTICAL)
        sizer_9      = wx.StaticBoxSizer(wx.StaticBox(self.panel, wx.ID_ANY,
                                                      "Plot special"), wx.VERTICAL)
        sizer_10     = wx.StaticBoxSizer(wx.StaticBox(self.panel, wx.ID_ANY,
                                                      "Special options"), wx.VERTICAL)
        sizer_5      = wx.BoxSizer(wx.VERTICAL)
        grid_sizer_5 = wx.FlexGridSizer(0, 3, 0, 0)
        grid_sizer_4 = wx.FlexGridSizer(3, 3, 0, 0)
        sizer_7      = wx.BoxSizer(wx.VERTICAL)
        sizer_16     = wx.StaticBoxSizer(wx.StaticBox(self.panel, wx.ID_ANY,
                                                      "Plot"), wx.HORIZONTAL)
        grid_sizer_6 = wx.FlexGridSizer(6, 2, 0, 0)
        sizer_11     = wx.FlexGridSizer(1, 2, 0, 0)
        sizer_15     = wx.StaticBoxSizer(wx.StaticBox(self.panel, wx.ID_ANY,
                                                      "Plot options"), wx.VERTICAL)
        sizer_12     = wx.StaticBoxSizer(wx.StaticBox(self.panel, wx.ID_ANY,
                                                      "Spectral res."), wx.VERTICAL)
        sizer_13     = wx.StaticBoxSizer(wx.StaticBox(self.panel, wx.ID_ANY,
                                                      "Target, date and DIT"), wx.HORIZONTAL)
        grid_sizer_8 = wx.FlexGridSizer(4, 3, 0, 0)
        sizer_3      = wx.StaticBoxSizer(wx.StaticBox(self.panel, wx.ID_ANY,
                                                      "Input path"), wx.VERTICAL)
        grid_sizer_3 = wx.FlexGridSizer(0, 2, 0, 0)
        sizer_14     = wx.BoxSizer(wx.HORIZONTAL)
        sizer_14.Add(self.btn_open_oifits, 0, 0, 0)
        sizer_14.Add(self.btn_load_oifits, 0, 0, 0)
        sizer_3.Add(sizer_14, 1, wx.EXPAND, 0)
        label_1      = wx.StaticText(self.panel, wx.ID_ANY, "File/folder")
        grid_sizer_3.Add(label_1, 0, wx.ALL, 3)
        grid_sizer_3.Add(self.tb_path, 0, 0, 0)
        sizer_3.Add(grid_sizer_3, 1, wx.ALL | wx.EXPAND, 3)
        sizer_7.Add(sizer_3, 1, wx.EXPAND, 0)
        label_29     = wx.StaticText(self.panel, wx.ID_ANY, "Target")
        grid_sizer_8.Add(label_29, 0, 0, 0)
        grid_sizer_8.Add(self.tb_target, 0, 0, 0)
        grid_sizer_8.Add((0, 0), 0, 0, 0)
        label_23     = wx.StaticText(self.panel, wx.ID_ANY, "Date")
        grid_sizer_8.Add(label_23, 0, 0, 0)
        grid_sizer_8.Add(self.tb_date, 0, 0, 0)
        label_24     = wx.StaticText(self.panel, wx.ID_ANY, "")
        grid_sizer_8.Add(label_24, 0, 0, 0)
        label_25     = wx.StaticText(self.panel, wx.ID_ANY, "DIT")
        grid_sizer_8.Add(label_25, 0, 0, 0)
        grid_sizer_8.Add(self.tb_DIT, 0, 0, 0)
        label_26     = wx.StaticText(self.panel, wx.ID_ANY, "s")
        grid_sizer_8.Add(label_26, 0, 0, 0)
        label_27     = wx.StaticText(self.panel, wx.ID_ANY, "DITrange")
        grid_sizer_8.Add(label_27, 0, 0, 0)
        grid_sizer_8.Add(self.tb_DIT_range, 0, 0, 0)
        label_28     = wx.StaticText(self.panel, wx.ID_ANY, "s")
        grid_sizer_8.Add(label_28, 0, 0, 0)
        sizer_13.Add(grid_sizer_8, 1, wx.ALL | wx.EXPAND, 3)
        sizer_7.Add(sizer_13, 1, wx.EXPAND, 0)
        sizer_12.Add(self.cb_disp_LOW, 0, 0, 0)
        sizer_12.Add(self.cb_disp_MED, 0, 0, 0)
        sizer_12.Add(self.cb_disp_HIGH, 0, 0, 0)
        sizer_11.Add(sizer_12, 1, wx.ALL | wx.EXPAND, 3)
        sizer_15.Add(self.cb_pl_OI_t, 0, 0, 0)
        sizer_15.Add(self.cb_pl_OI_wl, 0, 0, 0)
        sizer_15.Add((0, 0), 0, 0, 0)
        sizer_11.Add(sizer_15, 1, wx.ALL | wx.EXPAND, 3)
        sizer_7.Add(sizer_11, 1, wx.EXPAND, 0)
        grid_sizer_6.Add(self.btn_VISAMP, 0, 0, 0)
        grid_sizer_6.Add(self.btn_VIS2, 0, 0, 0)
        grid_sizer_6.Add(self.btn_VISPHI, 0, 0, 0)
        grid_sizer_6.Add(self.btn_VIS, 0, 0, 0)
        grid_sizer_6.Add(self.btn_T3AMP, 0, 0, 0)
        grid_sizer_6.Add(self.btn_TF2, 0, 0, 0)
        grid_sizer_6.Add(self.btn_T3PHI, 0, 0, 0)
        grid_sizer_6.Add(self.btn_TF, 0, 0, 0)
        grid_sizer_6.Add(self.btn_CFX2, 0, 0, 0)
        grid_sizer_6.Add(self.btn_FLUX, 0, 0, 0)
        grid_sizer_6.Add(self.btn_CFXAMP, 0, 0, 0)
        grid_sizer_6.Add((0, 0), 0, 0, 0)
        sizer_16.Add(grid_sizer_6, 1, wx.EXPAND, 0)
        sizer_7.Add(sizer_16, 1, wx.EXPAND, 0)
        grid_sizer_1.Add(sizer_7, 1, wx.EXPAND, 0)
        grid_sizer_4.Add(self.cb_b_J, 0, 0, 0)
        grid_sizer_4.Add(self.cb_b_H, 0, 0, 0)
        grid_sizer_4.Add(self.cb_b_K, 0, 0, 0)
        grid_sizer_4.Add(self.cb_b_L, 0, 0, 0)
        grid_sizer_4.Add(self.cb_b_M, 0, 0, 0)
        grid_sizer_4.Add(self.cb_b_N, 0, 0, 0)
        grid_sizer_4.Add(self.cb_b_LM, 0, 0, 0)
        grid_sizer_4.Add((0, 0), 0, 0, 0)
        grid_sizer_4.Add((0, 0), 0, 0, 0)
        sizer_4.Add(grid_sizer_4, 1, wx.EXPAND, 0)
        label_7      = wx.StaticText(self.panel, wx.ID_ANY, "Wl min")
        grid_sizer_5.Add(label_7, 0, 0, 0)
        grid_sizer_5.Add(self.tb_wl_min, 0, 0, 0)
        label_9      = wx.StaticText(self.panel, wx.ID_ANY, "um")
        grid_sizer_5.Add(label_9, 0, 0, 0)
        label_8      = wx.StaticText(self.panel, wx.ID_ANY, "Wl max")
        grid_sizer_5.Add(label_8, 0, 0, 0)
        grid_sizer_5.Add(self.tb_wl_max, 0, 0, 0)
        label_10     = wx.StaticText(self.panel, wx.ID_ANY, "um")
        grid_sizer_5.Add(label_10, 0, 0, 0)
        sizer_5.Add(grid_sizer_5, 1, wx.EXPAND, 0)
        sizer_5.Add(self.btn_def_wl, 0, 0, 0)
        sizer_4.Add(sizer_5, 1, wx.EXPAND, 0)
        sizer_10.Add(self.cb_errorbar, 0, 0, 0)
        sizer_4.Add(sizer_10, 1, wx.EXPAND, 0)
        sizer_9.Add(self.btn_VIS2_vs_freq, 0, 0, 0)
        sizer_9.Add(self.btn_VIS_vs_freq, 0, 0, 0)
        sizer_9.Add(self.btn_vis2_tf2_cp_vs_time, 0, 0, 0)
        sizer_9.Add(self.btn_vis_tf_cp_vs_time, 0, 0, 0)
        sizer_4.Add(sizer_9, 1, wx.EXPAND, 0)
        grid_sizer_1.Add(sizer_4, 1, wx.EXPAND, 0)
        self.panel.SetSizer(grid_sizer_1)
        sizer_1.Add(self.panel, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        self.Layout()
        # end wxGlade

    ###########################################################################

    def OnButtonClicked(self, e):
        plot_t_flag = self.cb_pl_OI_t.GetValue()
        plot_wl_flag = self.cb_pl_OI_wl.GetValue()
        plot_errorbar_flag = self.cb_errorbar.GetValue()
        self.DIT = float(self.tb_DIT.GetValue())
        self.DIT_range = float(self.tb_DIT_range.GetValue())
        band_mask = [self.cb_b_L.GetValue(),self.cb_b_M.GetValue(),self.cb_b_LM.GetValue(),self.cb_b_N.GetValue()]
        selected_bands = self.bands[band_mask].tolist()
        spectral_resolution_mask = [self.cb_disp_LOW.GetValue(),self.cb_disp_MED.GetValue(),self.cb_disp_HIGH.GetValue()]
        selected_spectral_resolutions = self.spectral_resolutions[spectral_resolution_mask].tolist()
        self.wl_min = float(self.tb_wl_min.GetValue())
        self.wl_max = float(self.tb_wl_max.GetValue())
        self.target_selected = self.tb_target.GetValue()
        eID = e.GetId()
        if eID < 13 or eID > 15:
            #first check if list of dictionaries is empty
            if len(self.list_of_dicts) == 1:
                if not self.list_of_dicts[0]:
                    self.statusbar.SetStatusText('Load OIFITS data.')
                    print ('Load OIFITS data.')
                    self.name_file = self.tb_path.GetValue()
                    self.OI_load_data(self.name_file)

            if len(self.list_of_dicts) > 1:
                if not self.target_selected.strip():
                    self.filtered_list_of_dicts = filter_oi_list(self.list_of_dicts, dates=[self.date],
                                   bands=selected_bands,
                                   spectral_resolutions=selected_spectral_resolutions,
                                   DIT_range=[self.DIT - self.DIT_range,
                                              self.DIT + self.DIT_range],
                                   targets=[])
                else:
                    self.filtered_list_of_dicts = filter_oi_list(self.list_of_dicts, dates=[self.date], bands=selected_bands, spectral_resolutions=selected_spectral_resolutions, DIT_range=[self.DIT - self.DIT_range, self.DIT + self.DIT_range], targets=[self.target_selected])
            else:
                self.filtered_list_of_dicts = self.list_of_dicts

                # Display of visibility amplitude
            if eID == 0:
                self.statusbar.SetStatusText('Show VISAMP.')
                print ('Show VISAMP.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts, [self.wl_min,self.wl_max], key="VIS", datatype="CFLUX",showvis=False,plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],key='VIS',
                                    datatype='CFLUX',showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of squared visibility
            elif eID == 1:
                self.statusbar.SetStatusText('Show VIS2.')
                print ('Show VIS2.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts, [self.wl_min,self.wl_max], key="VIS2", datatype="VIS2",showvis=False,plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],key='VIS2',
                                    datatype='VIS2',showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of differential visibility
            elif eID == 2:
                self.statusbar.SetStatusText('Show VISPHI.')
                print ('Show VISPHI.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts,
                                    [self.wl_min,self.wl_max], key="VIS",
                                    datatype="DPHI",showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],key='VIS',
                                    datatype='DPHI',showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of transfer function
            elif eID == 3:
                self.statusbar.SetStatusText('Show TF2.')
                print ('Show TF2.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts,
                                    [self.wl_min,self.wl_max], key="TF2",
                                    datatype="TF2",showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],key='TF2',
                                    datatype='TF2',showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of triple product amplitude
            elif eID == 4:
                self.statusbar.SetStatusText('Show T3AMP.')
                print ('Show T3AMP.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts,
                                    [self.wl_min,self.wl_max], key="T3",
                                    datatype="T3AMP",showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],key='T3',
                                    datatype='T3AMP',showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of flux
            elif eID == 5:
                self.statusbar.SetStatusText('Show FLUX.')
                print ('Show FLUX.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts,
                                    [self.wl_min,self.wl_max], key="FLUX",
                                    datatype="FLUX",showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],key='FLUX',
                                    datatype='FLUX',showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of closure phase
            elif eID == 6:
                self.statusbar.SetStatusText('Show T3PHI.')
                print ('Show T3PHI.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts,
                                    [self.wl_min,self.wl_max], key="T3",
                                    datatype="CLOS",showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],
                                    key='T3',datatype='CLOS',showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of differential visibility
            elif eID == 7:
                self.statusbar.SetStatusText('Show VIS.')
                print ('Show VIS.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts,
                                    [self.wl_min,self.wl_max], key="VIS2",
                                    datatype="VIS2",showvis=True,
                                    plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],key='VIS2',
                                    datatype='VIS2',showvis=True,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of transfer function
            elif eID == 8:
                self.statusbar.SetStatusText('Show TF.')
                print ('Show TF.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts,
                                    [self.wl_min, self.wl_max], key="TF2",
                                    datatype="TF2",
                                    showvis=True,
                                    plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0], key='TF2',
                                    datatype='TF2', showvis=True,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of visibilities as a function of spatial frequency
            elif eID == 9:
                self.statusbar.SetStatusText('Show VIS2 vs. spatial frequency.')
                print ('Show VIS2 vs. spatial frequency.')
                show_oi_vs_freq(self.filtered_list_of_dicts[0],log=False,
                                showvis=False)
            elif eID == 10:
                self.statusbar.SetStatusText('Show VIS vs. spatial frequency.')
                print ('Show VIS vs. spatial frequency.')
                show_oi_vs_freq(self.filtered_list_of_dicts[0],log=False,
                                showvis=True)
            elif eID == 11:
                self.statusbar.SetStatusText('Show VIS2, TF2, T3PHI vs time.')
                print ('Show VIS2, TF2, T3PHI vs time.')
                show_vis2_tf2_vs_time(self.filtered_list_of_dicts,
                                      [self.wl_min,self.wl_max], showvis=False, saveplots=False, output_path="",plot_errorbars=plot_errorbar_flag)
            elif eID == 12:
                self.statusbar.SetStatusText('Show VIS, TF, T3PHI vs time.')
                print ('Show VIS2, TF2, T3PHI vs time.')
                show_vis2_tf2_vs_time(self.filtered_list_of_dicts,
                                      [self.wl_min,self.wl_max], showvis=True,
                                      saveplots=False, output_path="",
                                      plot_errorbars=plot_errorbar_flag)
                # Display of squared correlated flux
            elif eID == 17:
                self.statusbar.SetStatusText('Show CFX2.')
                print ('Show CFX2.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts, [self.wl_min,self.wl_max], key="VIS2", datatype="CFX2",showvis=False,plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],key='VIS2',
                                    datatype='CFX2',showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
                # Display of correlated flux
            elif eID == 16:
                self.statusbar.SetStatusText('Show CFXAMP.')
                print ('Show CFXAMP.')
                if plot_t_flag == True:
                    show_oi_vs_time(self.filtered_list_of_dicts, [self.wl_min,self.wl_max], key="VIS", datatype="CFLUX",showvis=False,plot_errorbars=plot_errorbar_flag)
                elif plot_wl_flag == True:
                    show_oi_vs_wlen(self.filtered_list_of_dicts[0],key='VIS',
                                    datatype='CFLUX',showvis=False,
                                    plot_errorbars=plot_errorbar_flag)
        else:
            if eID == 13:
                self.statusbar.SetStatusText('Default wavelength range.')
                print ('Default wavelength range.')
                self.tb_wl_min.Clear()
                self.tb_wl_max.Clear()
                if self.cb_b_J.GetValue() == True:
                    self.wl_min = self.wl_min_def[0] #um, J,H,K,L,M,N,LM
                    self.wl_max = self.wl_max_def[0]
                if self.cb_b_H.GetValue() == True:
                    self.wl_min = self.wl_min_def[1]  # um, J,H,K,L,M,N,LM
                    self.wl_max = self.wl_max_def[1]
                if self.cb_b_K.GetValue() == True:
                    self.wl_min = self.wl_min_def[2]  # um, J,H,K,L,M,N,LM
                    self.wl_max = self.wl_max_def[2]
                if self.cb_b_M.GetValue() == True:
                    self.wl_min = self.wl_min_def[4] #um, J,H,K,L,M,N,LM
                    self.wl_max = self.wl_max_def[4]
                if self.cb_b_L.GetValue() == True:
                    self.wl_min = self.wl_min_def[3]
                    self.wl_max = self.wl_max_def[3]
                if self.cb_b_N.GetValue() == True:
                    self.wl_min = self.wl_min_def[5]
                    self.wl_max = self.wl_max_def[5]
                if self.cb_b_LM.GetValue() == True:
                    self.wl_min = self.wl_min_def[6]
                    self.wl_max = self.wl_max_def[6]
                self.tb_wl_min.write("%.2f" % (self.wl_min))
                self.tb_wl_max.write("%.2f" % (self.wl_max))
                # e.Skip()
            elif eID == 15:
                self.statusbar.SetStatusText('Open OIFITS data.')
                print ('Open OIFITS data.')
                print("Running file selector...")
                openFileDialog = mat_FileDialog(None, 'Open a file', "lmk,")
                if openFileDialog.ShowModal() == wx.ID_OK:
                    self.name_file = openFileDialog.GetPaths()[0]
                    print(self.name_file)
                else:
                    self.name_file = ""
                openFileDialog.Destroy()
                self.tb_path.SetValue(self.name_file)
                self.OI_load_data(self.name_file)
            elif eID == 14:
                self.statusbar.SetStatusText('Load OIFITS data.')
                print ('Load OIFITS data.')
                print(self.tb_path.GetValue())
                if not self.tb_path.GetValue():
                    openFileDialog = mat_FileDialog(None, 'Open a file', "lmk,")
                    if openFileDialog.ShowModal() == wx.ID_OK:
                        self.name_file = openFileDialog.GetPaths()[0]
                        self.OI_load_data(self.name_file)
                else:
                    self.name_file = self.tb_path.GetValue()
                    self.OI_load_data(self.name_file)

    ###########################################################################

    def OI_load_data(self,path):
        if os.path.isfile(path):
            self.dic = {}
            print("Reading file " + path + "...")
            self.dic = open_oi(path)
            self.list_of_dicts = [self.dic]
            # update the values in the form
            if self.dic['BAND'] == 'LM':
                self.cb_b_L.SetValue(False)
                self.cb_b_M.SetValue(False)
                self.cb_b_N.SetValue(False)
                self.cb_b_LM.SetValue(True)
                self.wl_min = self.wl_min_def[6] # um, J,H,K,L,M,N,LM
                self.wl_max = self.wl_max_def[6]
            elif self.dic['BAND'] == 'N':
                self.cb_b_L.SetValue(False)
                self.cb_b_M.SetValue(False)
                self.cb_b_N.SetValue(True)
                self.cb_b_LM.SetValue(False)
                self.wl_min = self.wl_min_def[5]
                self.wl_max = self.wl_max_def[5]

            if self.dic['DISP'] == 'LOW':
                self.cb_disp_LOW.SetValue(True)
                self.cb_disp_MED.SetValue(False)
                self.cb_disp_HIGH.SetValue(False)
            elif self.dic['DISP'] == 'MED':
                self.cb_disp_LOW.SetValue(False)
                self.cb_disp_MED.SetValue(True)
                self.cb_disp_HIGH.SetValue(False)
            elif self.dic['DISP'] == 'HIGH':
                self.cb_disp_LOW.SetValue(False)
                self.cb_disp_MED.SetValue(False)
                self.cb_disp_HIGH.SetValue(True)

            self.tb_DIT.SetValue("%.3f" % (self.dic['DIT']))
            self.tb_wl_min.Clear()
            self.tb_wl_max.Clear()
            self.tb_wl_min.write("%.2f" % (self.wl_min))
            self.tb_wl_max.write("%.2f" % (self.wl_max))
        elif os.path.isdir(path):
            name_dir = path
            self.dic = {}
            self.list_of_dicts = [{}]
            self.list_of_dicts = open_oi_dir(name_dir)
        else:
            "File/directory not found."
        self.date = self.list_of_dicts[0]['DATEOBS'][0:10]
        self.tb_date.SetValue(self.date)

###############################################################################

class OI_plotter(wx.App):
    def OnInit(self):
        self.frame = oi_data_select_frame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True

###############################################################################

if __name__ == '__main__':
    listArg = sys.argv
    name_file = []
    typePlot = "VIS2"
    for elt in listArg:
        if ('--help' in elt):
            print("Usage: mat_show_....py [--dir=start directory]")
            sys.exit(0)
        elif ('--typePlot' in elt):
            typePlot = elt.split("=")[1]
            # elif len(listArg) == 2:
            #     name_file = sys.argv[1]
            #     print(name_file)

    if typePlot == "VIS2":
        table = "VIS2"
    elif typePlot == "CLOS":
        table = "T3"
    elif typePlot == "TF2":
        typePlot = "TF2"
        table = "TF2"
    elif typePlot == "FLUX":
        table = "FLUX"
    elif typePlot == "DPHI":
        table = "VIS"

    app = OI_plotter(0)
    app.MainLoop()

    # app = wx.App()
    # if not name_file:
    #     print("No input name given, running file selector...")
    #     openFileDialog = mat_FileDialog(None, 'Open a file', "lmk,")
    #     if openFileDialog.ShowModal() == wx.ID_OK:
    #         name_file = openFileDialog.GetPaths()[0]
    #         print(name_file)
    #     openFileDialog.Destroy()
    # app.MainLoop()
    # app.Destroy()
    #
    # dic = {};
    # if os.path.isfile(name_file):
    #     print("Reading file " + name_file + "...")
    #     dic = open_oi(name_file)
    #     print("Plotting data " + name_file + "...")
    #     show_oi_vs_freq(dic)
    #     print("Plotting data " + name_file + "...")
    #     plt.figure()
    #     show_oi_vs_wlen(dic['FLUX'], dic['WLEN'], datatype='FLUX')
    #     print("Plotting data " + name_file + "...")
    #     plt.figure()
    #     wlen = dic['WLEN']
    #     print(wlen)
    # elif os.path.isdir(name_file):
    #     name_dir = name_file
    #     list_of_dicts = open_oi_dir(name_dir)
    #     filtered_list_of_dicts = filter_oi_list(list_of_dicts)
    #     show_vis2_tf2_vs_time(filtered_list_of_dicts, [3.5, 3.95])
    #     # show_oi_vs_time(filtered_list_of_dicts ,[3.5,3.95],key=table, datatype=typePlot)
