#!/bin/env python3.7.2
# Imports
import matplotlib.pylab as plt
import numpy as np
import MDAnalysis as mda

# unwrapping coordinates
# Input: cell is the boundary
#        curCoords is current location of atoms
#        prevCoords is previous location of atoms
# Output: Unwrapped Coordinates
def removePBC(cell, curCoords, prevCoords):
    unwrapCoords = curCoords.copy()
    dim = [cell[i] for i in range(3)]
    hbox = [0.5 * dim[i] for i in range(3)]
    natoms = len(curCoords)
    shiftx = np.zeros(natoms)
    shifty = np.zeros(natoms)
    shiftz = np.zeros(natoms)
    tmpdz = curCoords[:,2] - prevCoords[:,2]  # only consider z-direction at this time
    i = 0
    for dz in tmpdz:
        shift = 0
        if dz > hbox[2]:
            shift -= 1
            while (dz + shift * dim[2]) > hbox[2]:
                shift -= 1
        elif dz < -hbox[2]:
            shift += 1
            while (dz + shift * dim[2]) < -hbox[2]:
                shift += 1
        shiftz[i] = shift
        i += 1
    unwrapCoords[:,2] += (shiftz * dim[2])
    tmpdy = curCoords[:,1] - prevCoords[:,1]  # only consider y-direction at this time
    i = 0
    for dy in tmpdy:
        shift = 0
        if dy > hbox[1]:
            shift -= 1
            while (dy + shift * dim[1]) > hbox[1]:
                shift -= 1
        elif dy < -hbox[1]:
            shift += 1
            while (dy + shift * dim[1]) < -hbox[1]:
                shift += 1
        shifty[i] = shift
        i += 1
    unwrapCoords[:,1] += (shifty * dim[1])
    tmpdx = curCoords[:,0] - prevCoords[:,0]  # only consider x-direction at this time
    i = 0
    for dx in tmpdx:
        shift = 0
        if dx > hbox[0]:
            shift -= 1
            while (dx + shift * dim[0]) > hbox[0]:
                shift -= 1
        elif dx < -hbox[0]:
            shift += 1
            while (dx + shift * dim[0]) < -hbox[0]:
                shift += 1
        shiftx[i] = shift
        i += 1
    unwrapCoords[:,0] += (shiftx * dim[0])        
    return unwrapCoords

# Open Simulation
index_file = "cg_topol.tpr"
binary_file = "cg_traj.xtc"
uta = mda.Universe(index_file, binary_file)

# Get all atoms
aL = uta.select_atoms('all') 

# Start and end of unwrap
st = 10000 # start
en = 210000 # end

# Unwrap Coordinates
firstframe = True
with mda.Writer("cg_unwrap3.xtc", aL.n_atoms) as W:
    for ts in uta.trajectory:
        if ts.frame >= st:
            cell = ts.dimensions
            if firstframe:
                prevaL = aL.positions
                firstframe = False
            curaL = aL.positions
            tmpaL = removePBC(cell, curaL, prevaL)
            prevaL = tmpaL
            
            # change all of the positions of ts
            aL.positions = tmpaL

            # write to new file
            W.write(aL)
        if ts.frame % 1000 == 0:
            print("Frame {}".format(ts.frame))
        if ts.frame == en:
            break

