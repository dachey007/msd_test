import MDAnalysis as mda

traj = 'cg_unwrap.xtc'
uta = mda.Universe('cg_topol.tpr', traj)
aP = uta.select_atoms('type PF')
counter = 0
print(uta.trajectory[0].dimensions)
for atom in aP:
    for ts in uta.trajectory:
        #print(here)
        if ts.frame >= 0:
            print(atom.position)
            counter += 1
            if counter % 50 == 0:
                break
    if counter % 500 == 0:
        break

