#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

   Created on Wed Apr  5 10:18:07 2017
@author: ame, jvarga, fmillour

file selector for the MATISSE instrument.

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

#TODO: fix file type filter issues

#TODO: if a folder does not contain fits files, then don't create the
#      buffer file

# Import necessary files
from   libAutoPipeline import matisseType
import wx
import os
from   ObjectListView import ObjectListView, ColumnDefn, Filter
from   astropy.io import fits
import subprocess
#from   mat_showFitsHeader import mat_showFitsHeader
import distutils.spawn
import fnmatch
import sys
#import cPickle as pickle
import pickle

listArg = sys.argv
for elt in listArg:
    if ('--help' in elt):
        print ("Usage: mat_fileDialog.py [--dir=start directory]")
        sys.exit(0)

repBase = []
for elt in listArg:
    if ('--dir' in elt):
        item=elt.split('=')
        repBase=item[1]

# Set useful paths
fvpath    = distutils.spawn.find_executable("fv")
iconspath = os.path.join(os.path.dirname(__file__),"icons")
# Set buffer filename:
FileList_FileName='mat_fileDialog.pkl'

###############################################################################

class fileViewerKeyword:
    def __init__(self,headerkeyword=None,name=None,checkheader=None, checkheader_cases=None,function=None,source=None):
        self.headerkeyword     = headerkeyword
        self.name              = name
        self.checkheader       = checkheader
        self.checkheader_cases = checkheader_cases
        self.function          = function
        self.source            = source

    def evaluate(self,header=None,filename=None):
        res = ""
        if (self.function and self.source   == "header"):
            res = eval("{0}(header)".format(self.function))
        elif (self.function and self.source == "filename"):
            res = eval("{0}(filename)".format(self.function))
        elif self.headerkeyword and header:
            if self.checkheader:
               check = header[self.checkheader]
               for i in range(len(self.checkheader_cases)):
                   if check == self.checkheader_cases[i]:
                       res = header[self.headerkeyword[i]]
            else:
               res = header[self.headerkeyword]

        if (res == "PHOTO"):
            res = "SIPHOT";
        elif(res == "INTER"):
            res = "HISENS";

        return res

    def __str__(self):

        res="NAME={0}".format(self.name)
        if  self.headerkeyword:
            res+=" HEADERKEYWORD={0}".format(self.headerkeyword)
        if  self.checkheader:
            res+=" CHECKHEADER={0}".format(self.checkheader)
        if  self.checkheader_cases:
            res+=" CHECKHEADER8CASE={0}".format(self.checkheader_cases)
        if  self.function:
            res+=" FUNCTION={0}".format(self.function)
        if  self.source:
            res+=" SOURCE={0}".format(self.source)
        return res

###############################################################################

dict_keys = ['DATE-OBS', 'INSTRUME','OBJECT',
             'HIERARCH ESO OBS TARG NAME',
             'HIERARCH ESO DET CHIP NAME',
             'HIERARCH ESO DET CHIP TYPE',
             'HIERARCH ESO DET NDIT',
             'HIERARCH ESO DET SEQ1 DIT',
             'HIERARCH ESO DET DIT',
             'HIERARCH ESO INS PIL NAME',
             'HIERARCH ESO INS PIN NAME',
             'HIERARCH ESO INS DIL NAME',
             'HIERARCH ESO INS DIN NAME',
             'HIERARCH ESO PRO CATG',
             'HIERARCH ESO DPR CATG',
             'HIERARCH ESO DPR TYPE',
             'HIERARCH ESO DPR TECH',
             'HIERARCH ESO DET NAME',
             'HIERARCH ESO DET READ CURNAME',
             'HIERARCH ESO TPL START',
             'HIERARCH ESO INS PIL ID',
             'HIERARCH ESO INS PIN ID',
             'HIERARCH ESO INS DIL ID',
             'HIERARCH ESO INS DIN ID',
             'HIERARCH ESO INS POL ID',
             'HIERARCH ESO INS FIL ID',
             'HIERARCH ESO INS PON ID',
             'HIERARCH ESO INS FIN ID',
             'HIERARCH ESO DET SEQ1 PERIOD']

keywords=[]

keywords.append(fileViewerKeyword(
    headerkeyword="DATE-OBS",
    name="Date"))

keywords.append(fileViewerKeyword(
    headerkeyword = "HIERARCH ESO OBS TARG NAME",
    name          = "Target"))

keywords.append(fileViewerKeyword(
    function = "matisseType",
    source   = "header",
    name     = "DoCatg"))

keywords.append(fileViewerKeyword(
    headerkeyword = "HIERARCH ESO DET CHIP TYPE",
    name          = "Wavelength"))

keywords.append(fileViewerKeyword(
    headerkeyword = ["HIERARCH ESO DET SEQ1 DIT","HIERARCH ESO DET DIT"],
    checkheader       = "INSTRUME",
    checkheader_cases = ["MATISSE","AMBER"],
    name          = "DIT"))

keywords.append(fileViewerKeyword(
    headerkeyword = "HIERARCH ESO DET NDIT",
    name          = "NDIT"))

keywords.append(fileViewerKeyword(
    headerkeyword  = ["HIERARCH ESO INS PIL NAME","HIERARCH ESO INS PIN NAME"],
    checkheader       = "HIERARCH ESO DET CHIP NAME",
    checkheader_cases = ["HAWAII-2RG","AQUARIUS"],
    name              = "Mode"))

keywords.append(fileViewerKeyword(
    headerkeyword  = ["HIERARCH ESO INS DIL NAME","HIERARCH ESO INS DIN NAME"],
    checkheader       = "HIERARCH ESO DET CHIP NAME",
    checkheader_cases = ["HAWAII-2RG","AQUARIUS"],
    name              = "Resolution"))

keywords.append(fileViewerKeyword(
    headerkeyword= "HIERARCH ESO TPL START",
    name="TPL Start Date"))

#print(keywords)

###############################################################################

class identifyFile(object):
    def __init__(self,filename,folder):
        self.filename = filename
        self.folder   = folder

        self.header    = None
        self.isFits    = True
        self.isKnown   = True

        if os.path.isfile(folder+"/"+filename):
            self.isDir = False
        else:
            self.isDir = True

        if fnmatch.fnmatch(filename,"*.fits*"):
            #filename.endswith(".fits"):
            try:
                tmp_header  = fits.getheader( self.folder+"/"+self.filename)
                self.header = tmp_header.fromkeys(dict_keys)
                for ky in dict_keys:
                    try:
                        self.header[ky] = tmp_header[ky]
                    except:
                        self.header.remove(ky, ignore_missing=True, remove_all=False)

                #print(self.tmp_header['DATE-OBS'])
                #print(self.header['DATE-OBS'])

                #self.header=self.readFitsHeader( self.folder+"/"+self.filename)
                #print(self.header)
            except:
                self.isFits  = False
                self.isKnown = False
        else:
            self.isFits  = False
            self.isKnown = False

        for keywordi in keywords:
            try:
                self.__dict__[keywordi.name] = keywordi.evaluate(header=self.header,filename=folder+"/"+filename)

            except:
                self.__dict__[keywordi.name]=""
                self.isKnown = False

        if self.isDir:
            self.icon="Directory"
        else:
            if self.isKnown:
                try:
                    self.icon = self.header["INSTRUME"]+" File"
                except:
                    print("No INSTRUME keyword!")
                    self.icon="Fits File"
            elif self.isFits:
                self.icon="Fits File"
            else:
                self.icon="Normal File"

###############################################################################

matisseColor={
# Palette here http://colrd.com/palette/19308/
# Dark files
"DARK"           :wx.Colour(64,64,64),
"HOT_DARK"       :wx.Colour(192,192,192),
"OBSDARK"        :wx.Colour(64,64,64),
"DISTOR_HOTDARK" :wx.Colour(124,159,176),
"SPECTRA_HOTDARK":wx.Colour(124,159,176),
"KAPPA_HOTDARK"  :wx.Colour(128,128,255),
"REF_HOTDARK"    :wx.Colour(124,159,176),
"IM_COLD"        :wx.Colour(124,159,176),

# Sky files
"SKY_RAW"        :wx.Colour(64,255,255),

# Files with fringes (observations)
"TARGET_RAW"     :wx.Colour(154,255,136),
"CALIB_RAW"      :wx.Colour(154,255,136),

# Shift map
"SHIFT_MAP"      :wx.Colour(116,196,147),

# Bad Pixel file
"BADPIX"         :wx.Colour(201,74,83),

# NLM file
"NONLINEARITY"   :wx.Colour(101,56,125),

# Flat files
"FLAT"           :wx.Colour(228,191,128),
"OBSFLAT"        :wx.Colour(228,191,128),
"OBS_FLATFIELD"  :wx.Colour(228,191,128),
"IM_FLAT"        :wx.Colour(228,191,128),

# Distorsion, spectral cliabration files
"DISTOR_IMAGES"  :wx.Colour(0,220,0),
"SPECTRA_IMAGES" :wx.Colour(0,220,0),

# Kappa matrix
"KAPPA_SKY"      :wx.Colour(150,0,50),
"KAPPA_SRC"      :wx.Colour(50,50,0),
"KAPPA_OBJ"      :wx.Colour(150,50,0),

    # Files with fringes (lab)
"CALIB_SRC_RAW"  :wx.Colour(154,191,136),

# Other files
"IM_PERIODIC"    :wx.Colour(0,200,50),
"IM_REF"         :wx.Colour(150,0,50),

# Palette here http://colrd.com/image-dna/23291/
# Also picked up some colours from "La Cinq"
# Products
"CALIB_CAL"      :wx.Colour(201,194,175),

"OBJ_CORR_FLUX"  :wx.Colour(96,189,175),
"PHOT_BEAMS"     :wx.Colour(161,216,177),
"OI_OPDWVPO"     :wx.Colour(107,150,129),
#"OI_OPDWVPO"    :wx.Colour(237,152,294),

# oifits
"RAW_DPHASE"     :wx.Colour(251,141,144),
"RAW_VIS2"       :wx.Colour(255,160,46),
"RAW_CPHASE"     :wx.Colour(255,229,93),
"RAW_SPECTRUM"   :wx.Colour(51,88,180),
"TARGET_RAW_INT" :wx.Colour(237,152,255),

"UNKNOWN"        :wx.Colour(235,235,235)
}
###############################################################################

def FileImageGetter(matFile):
   return matFile.icon

###############################################################################

class dirButtons(wx.BoxSizer):

    def __init__(self, parent,path="",updateFunction=None):
        super(dirButtons, self).__init__(wx.HORIZONTAL)
        self.parent=parent
        self.setPath(path)
        self.updateFunction = updateFunction

    def setPath(self,path):
        self.path=path
        #self.dirs=[diri for diri in path.split("/") if diri !='']
        self.dirs=path.split("/")
        self.Clear()
        self.Layout()
        self.ButtonList=[]
        size=0
        for diri in self.dirs:
            buti = wx.Button(self.parent,label=diri,style=wx.BU_EXACTFIT)
            self.ButtonList.append(buti)
            self.Add(buti,flag=wx.LEFT)
            size+=buti.GetSize()[0]
            buti.Bind(wx.EVT_BUTTON, self.ButtonClicked)

        if sys.platform=='win32' or sys.platform=='linux2':
            self.AddSpacer(-size) #je ne sais pas pourquoi il faut ajouter ça pour que l'affichage soit correct
        self.Layout()

    def ButtonClicked(self,event):
        btn   = event.GetEventObject()
        label = btn.GetLabel()
        #self.Hide(btn)
        #self.Remove(btn)
        i=self.dirs.index(label)
        newDir= ''.join([diri+"/" for diri in self.dirs[0:i+1]])
        if self.updateFunction:
            self.updateFunction(newDir)

###############################################################################

class mat_FileDialog(wx.Dialog):

    def __init__(self, parent, title=None,defaultDir=None,defaultFile=None,wildcard=None,style=None,pos=None):
        super(mat_FileDialog, self).__init__(parent, title=title, size=(1400, 600))

        if defaultDir and os.path.isdir(defaultDir):
              self.dir=defaultDir
        else:
            self.dir=os.getcwd()

        self.InitUI()


        self.Centre()
        self.Show()
        self.path=""

    def InitUI(self):

        # Gobal window
        panel = wx.Panel(self)
        font  = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(9)

        # Structure the window vertically
        vbox  = wx.BoxSizer(wx.VERTICAL)
        vbox.AddSpacer(10)

        # Top arborescence
        hbox  = wx.BoxSizer(wx.HORIZONTAL)
        self.dirButtons = dirButtons(panel)
        hbox.Add(self.dirButtons)
        vbox.Add(hbox, proportion=0.1, flag=wx.LEFT|wx.RIGHT|wx.EXPAND,border=10)
        vbox.AddSpacer(10)


        # File browser with side panel
        splitter     = wx.SplitterWindow(panel)
        # Directories tree (left panel)
        self.dirTree = wx.GenericDirCtrl(splitter, style=wx.DIRCTRL_DIR_ONLY|wx.DIRCTRL_EDIT_LABELS)
        self.dirButtons.updateFunction=self.dirTree.SetPath
        if self.dir:
            self.dirTree.SetPath(self.dir)

        # List of files with colours (right panel)
        self.fileList = ObjectListView(splitter,wx.ID_ANY, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.fileList.AddNamedImages("Directory",    wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, wx.Size(16,16)))
        self.fileList.AddNamedImages("Normal File",  wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, wx.Size(16,16)))
        self.fileList.AddNamedImages("MATISSE File", wx.Bitmap(os.path.join(iconspath,"matisseSmall.ico")))
        self.fileList.AddNamedImages("AMBER File",   wx.Bitmap(os.path.join(iconspath,"amberSmall.ico")))
        self.fileList.AddNamedImages("GRAVITY File", wx.Bitmap(os.path.join(iconspath,"gravitySmall.ico")))
        self.fileList.AddNamedImages("Fits File",    wx.Bitmap(os.path.join(iconspath,"fitsSmall.ico")))
        cols=[ColumnDefn("Filename","left",300,"filename",minimumWidth=250,imageGetter=FileImageGetter),
              ColumnDefn("Type","left",150,"icon",minimumWidth=50)]
        cols.extend([ColumnDefn(keywordi.name,"left",75,keywordi.name,minimumWidth=50) for keywordi in keywords])
        self.fileList.SetColumns(cols)
        self.fileList.rowFormatter=self.setRowColor
        self.fileList.AutoSizeColumns()

        # sort columns by default to file name and type
        self.fileList.SortBy(2, ascending=True)
        #self.fileList.SetSortColumn(0, resortNow=True)
        #self.fileList.SortBy(0)
        #self.fileList.SortBy(1)
        #self.fileList.SetSortColumn(1, resortNow=True)
        #self.fileList.DoSort("Type", SortOrder.Ascending)
        splitter.SplitVertically(self.dirTree,self.fileList, sashPosition=350)
        splitter.SetMinimumPaneSize(200)
        vbox.Add(splitter, proportion=1, flag=wx.LEFT|wx.RIGHT|wx.EXPAND,border=10)

        vbox.AddSpacer(10)

        hbox2             = wx.BoxSizer(wx.HORIZONTAL)
        self.addDirButton = wx.Button(panel,label="+")
        hbox2.Add(self.addDirButton, proportion=0.1,flag=wx.LEFT|wx.RIGHT|wx.EXPAND)
        self.pathText  = wx.TextCtrl(panel,style=wx.TE_READONLY)
        hbox2.Add(self.pathText, proportion=1,flag=wx.LEFT|wx.RIGHT|wx.EXPAND)
        self.filterBox = wx.ComboBox(panel,choices=["All Files","Fits Files","Matisse Files"],style    = wx.CB_DROPDOWN|wx.CB_READONLY)
        self.filterBox.SetValue("All Files")
        hbox2.Add(self.filterBox, proportion=0.2,flag=wx.LEFT|wx.RIGHT|wx.EXPAND)
        self.cancelButton = wx.Button(panel,wx.ID_CANCEL)
        hbox2.Add(self.cancelButton,proportion=0.1,flag=wx.LEFT|wx.RIGHT|wx.EXPAND)
        self.okButton = wx.Button(panel,wx.ID_OK )
        hbox2.Add(self.okButton,proportion=0.1,flag=wx.LEFT|wx.RIGHT|wx.EXPAND)
        vbox.Add(hbox2, proportion=0.2, flag=wx.LEFT|wx.RIGHT|wx.EXPAND,border=10)
        vbox.AddSpacer(10)

        panel.SetSizer(vbox)

        tree = self.dirTree.GetTreeCtrl()
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.dirChanged,tree)
        self.Bind(wx.EVT_BUTTON, self.okClicked, self.okButton)
        self.Bind(wx.EVT_BUTTON, self.cancelClicked, self.cancelButton)
        self.Bind(wx.EVT_COMBOBOX, self.filterChanged, self.filterBox)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED,self.fileSelected,self.fileList)
        self.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK,self.fileListRightClicked,self.fileList)
        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED,self.fileListDoubleClicked,self.fileList)
        self.Bind(wx.EVT_BUTTON, self.addDirClicked, self.addDirButton)

        self.dirChanged()

        # Colour of text
    def setRowColor(self,listItem, data):

        if data.DoCatg:
            try:
                bkgcol=matisseColor[data.DoCatg]
                sm = sum(bkgcol)
                if sm > 2*255:
                    txtcol=wx.BLACK
                else:
                    txtcol=wx.WHITE

            except:
                txtcol=wx.BLACK
                bkgcol=matisseColor["UNKNOWN"]
        else:
            c=listItem.GetId() % 2
            bkgcol=wx.Colour(250+3*c,250+3*c,250+3*c)
            if data.isDir:
                txtcol=wx.Colour(0,0,0)
            else:
                txtcol=wx.Colour(180,180,180)
        listItem.SetTextColour(txtcol)
        listItem.SetBackgroundColour(bkgcol)

    def dirChanged(self,treeEvent=None):
       self.pathText.SetValue("")
       #newDir= self.dirTree.GetPath()
       self.dir=self.dirTree.GetPath().replace("\\","/")
       files=os.listdir(self.dir)
       matisseFileList=[]
       filt=self.filterBox.GetValue()

       #try to open the FileList_File in the directory
       #if os.path.isfile(self.dir + '/' + FileList_FileName):
       #print "----dirChanged-----"
       try:
           # if file exists, open it
           print ("Opening existing FileList_File...")
           FileList_File = open(self.dir + '/' + FileList_FileName, 'rb')
           #print "Load with pickle"
           matisseFileList = pickle.load(FileList_File)
           FileList_File.close()
           tmp_filename_list = []
           #print "Generate tmp_filename_list"
           for i in range(len(matisseFileList)):
               #print "   " + matisseFileList[i].filename
               tmp_filename_list.append(matisseFileList[i].filename)

           #synchronize matisseFileList with the actual directory listing
           #print "Generate sets"
           A = set(tmp_filename_list)
           B = set(files)
           # list of files present in the buffer but not in the current directory listing
           #print "delete_list"
           delete_list = list(A - B)
           delete_flag = 0
           if delete_list:
               #print "there are files to delete from list"
               #these files have been removed from the directory, so they should be removed from the file list
               for filei in delete_list:
                   #print "   " + filei
                   del_idx = delete_list.index(filei)
                   del matisseFileList[del_idx]
                   delete_flag = 1
           # list of files present in the directory but not present in the file list
           #print "add_list"
           add_list = list(B - A)
           add_flag = 0
           if add_list:
               #print "add new files to the file list"
               #add new files to the file list
               for filei in add_list:
                   #if os.path.isfile(self.dir+"/"+filei):
                       #print "   " + filei
                       matFile = identifyFile(filei,self.dir)
                       if filt == "All Files":
                           Append = True
                       elif filt == "Fits Files" and matFile.isFits:
                           Append = True
                       elif filt == "Matisse Files" and matFile.isMatisse:
                           Append = True
                       else:
                           Append = False
                       if Append:
                           matisseFileList.append(matFile)
                           add_flag = 1
           #if there were changes in the file list, overwrite the old list file
           if (delete_flag == 1 or add_flag == 1):
               #print "there were changes in the file list, overwrite the old list file"
               # save matisseFileList in binary file
               FileList_File = open(self.dir + '/' + FileList_FileName, 'wb')
               pickle.dump(matisseFileList, FileList_File,2)#pickle.HIGHEST_PROTOCOL)
               FileList_File.close()
       except (IOError,EOFError) as error:
           # the file does not exists, or is empty
           print ("FileList does not exist. Creating a new FileList_File...")
           #print "add new files to the file list"
           fitsInDir = False;
           for filei in files:
               #if os.path.isfile(self.dir+"/"+filei):
                   #print "   " + filei
                   matFile = identifyFile(filei,self.dir)
                   if filt == "All Files":
                       Append = True
                   elif filt == "Fits Files" and matFile.isFits:
                       Append = True
                   elif filt == "Matisse Files" and matFile.isMatisse:
                       Append = True
                   else:
                       Append = False
                   if Append:
                       matisseFileList.append(matFile)
                   if matFile.isFits:
                       fitsInDir = True

           if fitsInDir == True:
               #save matisseFileList in file
               #print "save matisseFileList in file"
               FileList_File = open(self.dir+'/'+ FileList_FileName, 'w+b')
               pickle.dump(matisseFileList, FileList_File)
               FileList_File.close()
           else:
               print ("No fits files, skipping...")

       self.fileList.SetObjects(matisseFileList)
       #flt = Filter.TextSearch(self, columns=self.fileList.columns[1], text="Fits File")
       #self.fileList.SetFilter(flt)
       self.fileList.AutoSizeColumns()

       for icol in range(len(keywords)+1):
           self.fileList.SetColumnWidth(icol,wx.LIST_AUTOSIZE)
           wc=self.fileList.GetColumnWidth(icol)
           if wc<self.fileList.columns[icol].minimumWidth:
               self.fileList.SetColumnWidth(icol,self.fileList.columns[icol].minimumWidth)

       self.dirButtons.setPath(self.dir)

    def fileSelected(self,event):
        nfiles=self.fileList.GetSelectedItemCount()
        itemNum=-1
        selectedItems=[]
        for ifile in range(nfiles):
            itemNum=self.fileList.GetNextSelected(itemNum)
            selectedItems.append(self.fileList.GetItem(itemNum).GetText())
        self.path=selectedItems
        txt=""
        for itemi in selectedItems:
            txt=txt+itemi+" "
        self.pathText.SetValue(txt)


    def fileListDoubleClicked(self,event):
        # Change directory if it is a directory
        if identifyFile(self.path[0],self.dirTree.GetPath()).isDir:
            print ("Change to "+self.dirTree.GetPath()+'/'+self.path[0])
            self.dirTree.SetPath(self.dirTree.GetPath()+'/'+self.path[0])
        # or select file and exist if it is a file
        else:
            print("Select file "+self.path[0])
            self.EndModal(wx.ID_OK)

    def fileListRightClicked(self,event):
        print ("rightclicked")
        menu = wx.Menu()
        menu.Append( 0, "Show Header" )
        wx.EVT_MENU( menu, 0, self.showHeader)
        menu.Append( 1, "Show RAW_DATA")
        wx.EVT_MENU( menu, 1, self.showRawData)
        menu.Append( 2, "Show OIFITS")
        wx.EVT_MENU( menu, 2, self.showOiFits)
        menu.Append( 3, "Open with fv" )
        wx.EVT_MENU( menu, 3, self.openWithFv)
        menu.Append(4, "Copy list to clipboard")
        wx.EVT_MENU(menu, 4, self.copyListToClipboard)

        self.fileList.PopupMenu( menu, event.GetPoint())

    def okClicked(self,event):
        self.EndModal(wx.ID_OK)

    def cancelClicked(self,event):
         self.EndModal(wx.ID_CANCEL)

    def addDirClicked(self,event):

        os.makedirs(self.dir+"/"+"newDir")
        self.dirTree.ReCreateTree()
        self.dirTree.ExpandPath(self.dir+"/"+"newDir")

    def filterChanged(self,event):
        self.dirChanged()

    def GetPaths(self):
        return [self.dir+'/'+pathi for pathi in self.path]

    def showHeader(self,event):
        print(("show header {0}".format(self.GetPaths())))
        for filei in self.path:
             if filei.endswith('.fits'):
                 print((self.dir+'/'+filei))
                 mat_showFitsHeader(self.dirTree.GetPath()+'/'+filei)

    def showRawData(self,event):
        print(("show RAW_DATA {0}".format(self.GetPaths())))
        import mat_show_rawdata as shra
        for filei in self.path:
            if filei.endswith('.fits'):
                name_file = self.dirTree.GetPath()+'/'+filei
                #print(("Reading file "+name_file+"..."))
                #dic = shra.open_mat(name_file)
                print(("Plotting data "+name_file+"..."))
                #shra.show_mat(dic)
                shra.mat_show_rawdata(name_file)
    def showOiFits(self,event):
        print(("show OIFITS {0}".format(self.GetPaths())))
        for filei in self.path:
            if filei.endswith('.fits'):
                subprocess.Popen(["python mat_show_oifits.py",self.dirTree.GetPath()+'/'+filei])

    def openWithFv(self,event):
        print(("Open with fv {0}".format(self.GetPaths())))
        for filei in self.path:
            if filei.endswith('.fits'):
                subprocess.Popen([fvpath,self.dirTree.GetPath()+'/'+filei])

    def copyListToClipboard(self,event):
        print ("Copy list to clipboard")
        self.dataObj = wx.TextDataObject()
        # Get the column header text
        h = []
        for n in range(self.fileList.GetColumnCount()):
            col = self.fileList.GetColumn(n)
            h.append(col.GetText())
        header = ';'.join(h)
        # Get all the values of the given rows into multi-list
        rows = self.fileList._GetValuesAsMultiList(self.fileList.GetObjects())
        # Make a text version of the values
        lines = [";".join(x) for x in rows]
        txt = '#' + header + '\n' + "\n".join(lines) + "\n"
        self.dataObj.SetText(txt)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(self.dataObj)
            wx.TheClipboard.Close()
        else:
            print ("Unable to open the clipboard")

###############################################################################

if __name__ == '__main__':
    app = wx.App()
    openFileDialog = mat_FileDialog(None, 'Open a file',repBase)
    if openFileDialog.ShowModal() == wx.ID_OK:
        print((openFileDialog.GetPaths()))
    openFileDialog.Destroy()
    app.MainLoop()
    app.Destroy()
