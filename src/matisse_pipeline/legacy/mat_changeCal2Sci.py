#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Fri Nov 16 13:50:15 2018
@author: fmillour

Change a science to calibrator and vice versa

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

from   astropy.io import fits as fits
import wx
import numpy as np
from mat_fileDialog import mat_FileDialog
from mat_show_oifits import open_oi,open_oi_dir,filter_oi_list



class changeKey(wx.Frame):
    def __init__(self, parent, title):
        super(changeKey, self).__init__(parent, title = title,size = (400,350))
        self.dir = "";
        self.panel = wx.Panel(self)
        self.vbox  = wx.BoxSizer(wx.VERTICAL)

        # Load files button
        self.loadfilesbtn = wx.Button(self.panel,-1,"Load Files")
        self.vbox.Add(self.loadfilesbtn,0,wx.ALIGN_CENTER)
        self.loadfilesbtn.Bind(wx.EVT_BUTTON,self.OnLoadFiles)
        #Exit button
        self.exitbtn = wx.Button(self.panel , -1, "Exit")
        self.vbox.Add(self.exitbtn,0,wx.ALIGN_CENTER)
        self.exitbtn.Bind(wx.EVT_BUTTON,self.OnExit)

        self.vbox.AddSpacer(20)

        self.panel.SetSizer(self.vbox)
        # Plot the GUI
        self.Centre()
        self.Show()
        self.Fit()

    def OnExit(self, event):
        print("Exiting...")
        #self.CloseEvent()
        #self.Destroy()
        app.MainLoop()
        #app.Destroy()

    def OnClicked(self, event):
        btn = event.GetEventObject().GetLabel()
        print ("Label of pressed button = ",btn)

    def OnLoadFiles(self,event):
        openFileDialog = mat_FileDialog(None, 'Select a directory',self.dir)
        print("Loading files...")
        if openFileDialog.ShowModal() == wx.ID_OK:
            dirToOpen = openFileDialog.GetPaths()[0]
        print(dirToOpen)
        # Get list of files, target names, and data types
        self.dic = open_oi_dir(dirToOpen)

        self.getUniques(self.dic)

        self.Centre()
        self.Show()
        self.Fit()

        print("Done!")

    def getUniques(self,dic):
        listofstuff = []
        for idat in dic:
            obj    = idat['TARGET']
            typ    = idat['HDR']['ESO PRO CATG']
            caract = obj+';'+typ
            print(obj+';'+typ)
            listofstuff = np.append(listofstuff, caract)

        self.redlist = set(listofstuff)
        self.buildListOfButtons()

    def buildListOfButtons(self):
        self.names = [mychar.split(";")[0] for mychar in self.redlist]
        self.iscal = [mychar.split(";")[1] for mychar in self.redlist]
        print(self.names)
        print(self.iscal)

        # Define the main buttons with the file selection here
        # input files part
        for idx,typi in enumerate(self.names):
            if self.iscal[idx] == 'CALIB_RAW_INT':
                color='red'
            else:
                color='green'

            self.__dict__[typi] = but(self,self.panel, typi, color)
            self.vbox.Add(self.__dict__[typi],border=20,flag=wx.LEFT|wx.RIGHT|wx.EXPAND)
            self.panel.SetSizer(self.vbox)
        self.btns = [self.__dict__[typi] for typi in self.names]
        self.vbox.AddSpacer(20)


        # Load files button
        self.applybtn = wx.Button(self.panel,-1,"Apply!")
        self.vbox.Add(self.applybtn,0,wx.ALIGN_CENTER)
        self.applybtn.Bind(wx.EVT_BUTTON,self.OnApply)
        self.panel.SetSizer(self.vbox)

        self.Centre()
        self.Show()
        self.Fit()

    def OnApply(self, event):
        dic = self.dic;
        self.names = [mychar.split(";")[0] for mychar in self.redlist]
        self.iscal = [mychar.split(";")[1] for mychar in self.redlist]
        for idat in dic:
        # Loop on files
            filetype = idat['HDR']['ESO PRO CATG']
            targname = idat['TARGET']
            filename = idat['file']

            comparname = np.nonzero(np.in1d(self.names, targname))

            new_color = self.__dict__[targname].color
            if new_color == 'green':
                new_type = 'TARGET_RAW_INT'
            elif new_color == 'red':
                new_type = 'CALIB_RAW_INT'
            else:
                print('ERROR!!!!!')

            print(filename)
            print(targname)
            print(filetype,new_type)
            if new_type != filetype:
                print('updated value for file. modifying...')
                fits.setval(filename, 'ESO PRO CATG', value=new_type)
            print('----')

class but(wx.BoxSizer):
    def __init__(self, parent, panel, butName, color):
        super(but, self).__init__(wx.VERTICAL)
        self.parent = parent
        self.color  = color;
        self.btn     = wx.Button     (panel, label=butName)
        self.btn.SetBackgroundColour(color)
        self.Add(self.btn,0,wx.ALIGN_CENTER)
        self.btn.Bind(    wx.EVT_BUTTON, self.onButton)

    #----------------------------------------------------------------------
    def onButton(self, event):
        if self.color == 'red':
            self.color = 'green'
        else:
            self.color = 'red'

        self.btn.SetBackgroundColour(self.color)

app = wx.App()
changeKey(None,  'Cal 2 Sci modifier')
app.MainLoop()
app.Destroy()
