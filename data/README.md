# Prepare the training data

## Download the normalized training data

Download the [training data](https://drive.google.com/open?id=0B720TgBOSgGRN01VbXcwOXMwMkE), and 
[training patch list](https://drive.google.com/open?id=0B720TgBOSgGRY2RvVmVET2Q1WW8). 
Upacking files like this structure:
```
+data
  +training
    +HG
    +LG
  +training_list
    -trainval-balanced.txt
    -trainval.txt
```

## Training list
For each row in training list, it gives the sampleID, index and label for the indexed pixel.
```
#ID      x    y    z    label
HG/0005  70   95   128  0
HG/0004  77   117  137  0
...
```
You can construct training batches (data pathch and labels) according to this list within your own data_loader.

## Generate hdf5 file
Use create_h5.py to generate hdf5 file. Run "python create_h5.py -h" for usage.
```
python ./create_h5.py --data_dir=/path/to/data --output_path=/path/to/h5_file
```
How to use:
```
import h5py
f = h5py.File('training.h5','r')
img_patch = f['HG/0001'][:, x-16:x+16+1, y-16:y+16+1, z] #sample a 33x33 patch
```
