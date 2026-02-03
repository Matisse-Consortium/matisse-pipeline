#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:41:41 2018

test

@author: fmillour
"""

import numpy as np
from numpy.linalg import eig, inv
import scipy
import wx
import sys
from scipy.spatial    import Voronoi, voronoi_plot_2d
from scipy.spatial    import Delaunay
from   matplotlib     import pyplot as plt
from   astropy.io     import fits as fits
from libShowOifits    import open_oi
from mat_fileDialog   import mat_FileDialog, identifyFile
from shapely.geometry import MultiPoint, Point, Polygon

###############################################################################

def open_hdr(oi_file):
    try:
        hdu = fits.open(oi_file)
    except IOError:
        print(("Unable to read fits file: " + oi_file))
        return {}

    hdr = hdu[0].header

    print(hdr)

    return hdr

###############################################################################

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

###############################################################################

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

###############################################################################

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

###############################################################################

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

###############################################################################

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

###############################################################################

def get_UV(file):

    BX = []
    BY = []

    res = open_hdr(file)
    try:
        instru = res['INSTRUME']
    except:
        try:
            instru = res['HIERARCH ESO INS MODE']
        except:
            print('error, unknown instrument!')

    print(instru)


    if instru == 'MATISSE':
        ntels = 4;
    elif instru == 'MIDI':
        ntels = 2;
    else:
        ntels = res['HIERARCH ESO DET NTEL']

    #read in priority the keywords
    try:
        base=0;
        for i in np.arange(1,ntels+1):
            for j in np.arange(i+1,ntels+1):
                tel1 = i;
                tel2 = j;
                base+=1;

                blen = (res['HIERARCH ESO ISS PBL'+str(tel1)+str(tel2)+' START']+res['HIERARCH ESO ISS PBL'+str(tel1)+str(tel2)+' END'])/2
                bang = (res['HIERARCH ESO ISS PBLA'+str(tel1)+str(tel2)+' START']+res['HIERARCH ESO ISS PBLA'+str(tel1)+str(tel2)+' END'])/2

                bx = blen * np.sin(bang * np.pi / 180.)
                by = blen * np.cos(bang * np.pi / 180.)

                BX = np.append(BX, bx)
                BY = np.append(BY, by)
    # otherwise look into the OI_VIS2 table directly
    except:
        print('No PBL keywords. taking a look into the OI_VIS2 table...')
        hdu = fits.open(file)
        print(hdu['OI_VIS2'])

    return BX,BY

###############################################################################

def get_UVs(files):
    BX = []
    BY = []

    for file in files:
        print(file)
        bx, by = get_UV(file)

        BX = np.append(BX, bx)
        BY = np.append(BY, by)

    return BX, BY;

###############################################################################

def plot_UV(BX, BY, marker='o', markersize=6, color="red"):
    plt.axis('equal')
    for i,base in enumerate(BX):

        plt.plot(base,BY[i], marker=marker, markersize=markersize, color=color)
        plt.plot(-base,-BY[i], marker='o', markersize=markersize-3, color=color)

    plt.plot(0,0, marker='+', markersize=markersize, color=color)

###############################################################################

# Enter your list of files (if you really want it!)
#files = [u'C:/DATA/WR104/2018-07-17T064751_WR104_IR-N_IN.fits', u'C:/DATA/WR104/2018-07-17T064751_WR104_IR-LM_IN.fits']
files = []

###############################################################################

if not files:
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

    if __name__ == '__main__':
        app = wx.App()
        openFileDialog = mat_FileDialog(None, 'Open a file',repBase)
        if openFileDialog.ShowModal() == wx.ID_OK:
            files = openFileDialog.GetPaths()
        openFileDialog.Destroy()
        app.MainLoop()
        app.Destroy()

###############################################################################

plt.figure(1)
plt.clf();
plt.subplot(3,3,1)

BX, BY = get_UVs(files)
plot_UV(BX, BY)

###############################################################################



BXA = np.append(BX, -BX)
BYA = np.append(BY, -BY)

BLDST = [];
BLANG = [];
for idx1,bas1 in enumerate(BXA):
    for idx2 in np.arange(idx1+1,len(BXA)):
        #if idx1 != idx2:
            #print(idx1,idx2)
        BLDST = np.append(BLDST, np.linalg.norm([BXA[idx1]-BXA[idx2],
                                                 BYA[idx1]-BYA[idx2]]))
        BLANG = np.append(BLANG, 180/np.pi*np.arctan2(BXA[idx1]-BXA[idx2],
                                                      BYA[idx1]-BYA[idx2]))
           #        print(bdist)




blen = np.linalg.norm([BX,BY],axis=0)
bmax = np.max(np.linalg.norm([BXA,BYA],axis=0))

###############################################################################
# histogram of baselines lengths

plt.subplot(3,3,4)
n, bins, patches = plt.hist(x=blen, bins=int(np.max(blen))+1, range=[0,bmax],
                            rwidth=0.85,
                            color='#0504aa')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value (m)')
plt.ylabel('Frequency')
plt.title('Histogram of bases')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=maxfreq + 1)

###############################################################################
# histogram of baselines angles

plt.subplot(3,3,5)
n, bins, patches = plt.hist(x=180/np.pi*np.arctan2(BXA,BYA), bins=int(np.max(blen))+1, rwidth=0.85,
                            color='#0504aa', orientation='horizontal')
plt.grid(axis='y', alpha=0.75)
plt.ylabel('Value (degrees)')
plt.xlabel('Frequency')
plt.title('Histogram of angles')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.xlim(xmax=maxfreq + 1)
plt.ylim(ymax=181,ymin=-1)

###############################################################################
# 2D histogram of baselines vectors

plt.subplot(3,3,6)
toto=plt.hist2d(np.linalg.norm([BXA,BYA],axis=0),180/np.pi*np.arctan2(BXA,BYA),
                              bins=int(np.max(blen))+1,range=[[0,bmax],
                                               [0,180]])


plt.tight_layout()

###############################################################################
# histogram of baselines separations

plt.subplot(3,3,7)
n, bins, patches = plt.hist(x=BLDST, bins=int(np.max(blen))+1, range=[0,bmax],
                            color='#0504aa',
                            rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value (m)')
plt.ylabel('Frequency')
plt.title('Histogram of bases differences')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax= maxfreq + 1)

###############################################################################
# histogram of baselines separation angles

plt.subplot(3,3,8)
n, bins, patches = plt.hist(x=BLANG, bins=int(np.max(blen))+1, color='#0504aa',
                            rwidth=0.85, orientation='horizontal')
plt.grid(axis='y', alpha=0.75)
plt.ylabel('Value (degrees)')
plt.xlabel('Frequency')
plt.title('Histogram of angles differences')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.xlim(xmax= maxfreq + 1)
plt.ylim(ymax=181,ymin=0)

###############################################################################
# 2D histogram of baselines separation vectors

plt.subplot(3,3,9)
toto=plt.hist2d(BLDST,BLANG,bins=int(np.max(blen))+1,range=[[0,np.max(np.linalg.norm([BXA,BYA],axis=0))],[0,180]])

###############################################################################
# Voronoi stuff!

plt.subplot(3,3,2)
plt.axis('equal')

# Build points with a duplicate center point (so that it splits the center cell in 2)
points = np.transpose([np.append(BXA,[0.01,-0.01]),np.append(BYA,[0,0])])
# Create Voronoi diagram out of the points, and clip infinite cells
vor = Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor)

# Get the outer convex shape (hull) englobing all points
pts = MultiPoint([Point(i) for i in points])
mask0 = pts.convex_hull

# expand hull with a factor 1.2
coord = list(mask0.exterior.coords)
for icoord,elem in enumerate(coord):
    elem = tuple([1.2 * x for x in elem])
    coord[icoord] = elem
mask1 = Polygon(coord)

# Fit an ellipse to the convex hull
coord = list(mask1.exterior.coords)
xp=[]
yp=[]
for icoord,elem in enumerate(coord):
    xp=np.append(xp,elem[0])
    yp=np.append(yp,elem[1])

a      = fitEllipse(xp,yp)
center = ellipse_center(a)
phi    = ellipse_angle_of_rotation2(a)
axes   = ellipse_axis_length(a)

# Plot the corresponding ellipse and define the mask as it
arc = 2.0
R = np.arange(0,arc*np.pi, 0.01*np.pi)
a, b = axes
xx = a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
yy = a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
plt.plot(xx,yy)
mask = Polygon(np.transpose([1.*xx,1.*yy]))

# Clip big cells to the ellipse and calculate all cells area
new_vertices = []
AREA = [];
for region in regions:
    polygon = vertices[region]
    shape = list(polygon.shape)
    shape[0] += 1
    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
    new_vertices.append(poly)
    area = Polygon(poly).area
    AREA = np.append(AREA,area)

# Fill cells with colors: green for small cells, and red for big ones
for region in regions:
    polygon = vertices[region]
    shape = list(polygon.shape)
    shape[0] += 1
    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
    area = Polygon(poly).area
    plt.fill(*zip(*poly), alpha=1, c=[area / np.max(AREA), 1 - area / np.max(AREA),0])

# Plot the UV coverage on top of the voronoi cells.
plot_UV(BX, BY,color='black',markersize=4)
#plt.plot(points[:,0], points[:,1], 'ko')

plt.title("OK for imaging?")
plt.show()




#tri = Delaunay(points)
#plt.triplot(points[:,0], points[:,1], tri.simplices)
