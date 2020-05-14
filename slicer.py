import MDAnalysis


# Start Universe
traj_old, bin_old = 'cg_topol.tpr', 'cg_unwrap2.xtc'
u = MDAnalysis.Universe(traj_old, bin_old)
aL = u.select_atoms("all")

print('starting slice')
traj_new, bin_new = 'slice_5ns.tpr', 'slice_100ns.xtc'
#Wr = TrajectoryWriter(traj_new, aL.n_atoms)
with MDAnalysis.Writer(bin_new, aL.n_atoms) as W:
    counter = 0
    for ts in u.trajectory[:100001]:
        W.write(aL)
        counter += 1
        if counter % 1000 == 0:
            print(counter)
print('done')
