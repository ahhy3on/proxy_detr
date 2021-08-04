from glob import glob
with open('trainset_voc.txt','w') as f:
    for year in [2007,2012]:
        paths = glob('./data/VOCdevkit/VOC'+str(year)+'/Annotations/*.xml')
        print('./data/VOCdevkit/VOC'+str(year)+'/Annotations/*.xml')
        print(paths)
        for p in paths:
            f.write(p)
            f.write('\n')

