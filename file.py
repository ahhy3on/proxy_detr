import glob
from PIL import Image
import random
paths = glob.glob('./cmes_new/xmls/**/*.xml')
print(paths[0])
random.shuffle(paths)
print(paths[0])
random.shuffle(paths)
print(paths[0])

train_paths = paths[:int(0.9*len(paths))]
val_paths = paths[int(0.9*len(paths)):]
with open('train_paths_skt_box.txt','w') as f:
    for p in train_paths:
        f.write(p)
        f.write('\n')

with open('val_paths_skt_box.txt','w') as f:
    for p in val_paths:
        f.write(p)
        f.write('\n')
    #temp = Image.open(p)
    #print(temp.size)
