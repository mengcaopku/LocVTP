import h5py

# Modify two terms
file_num = 8
file_name = "anetTextEncFixVideoScratchTSN_c3d_features.hdf5"

fout = h5py.File(file_name, 'w')

for idx in range(file_num):
    in_file = "{}_{}".format(str(idx), file_name)
    print("processing {}".format(in_file))
    with h5py.File(in_file, 'r') as fin: 
        for vid in fin.keys():
            if vid not in fout.keys(): # Avoid duplicate assignment
                feat = fin[vid]['c3d_features'][:]
                fgroup = fout.create_group(vid)
                fgroup.create_dataset('c3d_features', data=feat)

f = h5py.File(file_name, 'r')
print("fout contains {} video features.".format(len(f.keys())))
