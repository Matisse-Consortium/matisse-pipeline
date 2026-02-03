import sys
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess


def file_is_empty(path):
    return os.stat(path).st_size==0

def getSkyFile(filename):
    h=fits.open(filename)
    print(filename)
    splitt=filename.split('/')
    if len(splitt)==1:
        c=os.getcwd()
    else:
        c='/'.join(splitt[0:-1])

    tpl=h[0].header['HIERARCH ESO TPL START']
    detec=h[0].header['HIERARCH ESO DET CHIP NAME']
    cmd='dfits '+c+'/*.fits | fitsort dpr.type tpl.start det.chip.name | grep '+detec+'| grep '+tpl+' | grep SKY'
    test=os.system(cmd)
    if test==256:
        print('missing sky for this unchopped acquisiiton')
        print('can not continue')
        sys.exit(0)

    else:
         p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
         out, err = p.communicate()
         out=out.decode("utf-8")
         print(out)
         print(out.split('\n')[0])
         print(out.split('\n')[0].split('\t')[0])
         skyfile=(out.split('\n')[0].split('\t')[0])

    return skyfile




def mat_showFluxVsTime(filename):
    h=fits.open(filename)
    chopchop=h[0].header['HIERARCH ESO ISS CHOP ST']
    if chopchop=='T':
        print('Chopping case')
        skyfilename=""
        unchop=False
    else:
        print('UnChopping case')
        skyfilename=getSkyFile(filename)
        unchop=True
        print('Sky file used:',skyfilename)
    doSky = (skyfilename!="" or unchop==True)

    d=fits.open(filename)

    if doSky:
        dsky=fits.open(skyfilename)

    tartype=d['IMAGING_DATA'].data['TARTYP']
    tt=(tartype=='T')*1+(tartype=='U')*0.5
    targetframes=np.argwhere(tt==1)
    skyframes=np.argwhere(tt==0)


    colors=[]
    symsize=[]
    for i in range(len(tt)):
        if (tt[i]==0):
            colors.append('red')
            symsize.append(20)
        elif (tt[i]==0.5):
            colors.append('#00FF00')
            symsize.append(20)
        else:
            colors.append('blue')
            symsize.append(20)

    phot=[]
    flux=[]
    stddev=[]
    meanflux=[]

    h=d[0].header

    mode = h['ESO INS MODE']
    if (mode.strip() =="HYBRID" ):
        if (h['ESO DET NAME'].strip()=="MATISSE-LM") :
            mode = "SIPHOT"
        else:
            mode = "HISENS"

    if (mode.strip() =="IMAGING" ):
        mode="SIPHOT"


    print('mode ',h['ESO INS MODE'])
    print('detecteur ',h['ESO DET NAME'])

    if mode== "HISENS":
        regions=[5]
    else :
        regions = [9,10,11,12,13]

    nregions=len(regions)

    for iregion in regions:
        photi=d['IMAGING_DATA'].data['DATA{0}'.format(iregion)]*1.0
        if doSky:
            skyi=dsky['IMAGING_DATA'].data['DATA{0}'.format(iregion)]*1.0
            meamskyi=np.mean(skyi,axis=0)
            photi=photi-meamskyi
        phot.append(photi)

        fluxi=np.sum(np.sum(photi,axis=1),axis=1)
        if unchop:
            fluxi= np.take(fluxi,targetframes)
        flux.append(fluxi)
        stddev.append(np.std(fluxi))
        meanflux.append(np.mean(fluxi))
    print(np.array(stddev)/np.array(meanflux))
    print(meanflux)
    col=["violet","skyblue","salmon","peru","olive"]
    if mode == "SIPHOT":
        #renormalization of the interferometric channel
        mult=[1,1,0.23,1,1]
    else:
        mult=[1]
    for i in range(nregions):
        flux[i]=flux[i]*mult[i]
        plt.plot(flux[i],marker=".",color=col[i])


    mini=np.min(flux)
    maxi=np.max(flux)
    x=np.arange(len(tt))
    y=tt*(maxi-mini)+mini
    plt.scatter(x,y, color=colors)

    plt.show()



if  __name__== '__main__' :


    arg=sys.argv

    if (arg[1]=="--help" or arg[1]== "-h"):
        print( "mat_fringe_flux_plot script to visualize flux vs time on fringes exposures")
        print( "Usage : filename [-options]")
        print( "options :")
        print( " --sky skyfilename")
        print( " --unchop")
    else:
        filename=arg[1]

        mat_showFluxVsTime(filename)
