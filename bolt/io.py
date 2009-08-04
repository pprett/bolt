import numpy as np
import gzip

def load(filename):
    if filename.endswith('.npz'):
        return loadNpz(filename)
    else:
        return loadDat(filename)

def loadNpz(filename):
    print 'loading numpy archive'
    f=np.load(filename)
    examples = f['arr_0']
    labels = f['arr_1']
    return examples,labels

def loadDat(filename):
    labels = []
    examples = []
    global_max = -1
    t_pair = np.dtype("u4,f4")
    if filename.endswith('gz'):
        f=gzip.open(filename,'r')
    else:
        f=open(filename,'r')
    try:
        for i,line in enumerate(f):
            tokens = line.rstrip().split()
            label = float(tokens[0])
            labels.append(label)
            tokens=[(int(t[0]),float(t[1]))
                    for t in (t.split(':')
                              for t in tokens[1:] if t != '')]
            a=np.array(tokens, dtype=t_pair)
            local_max = 0.0
            if a.shape[0]>0:
                local_max = a['f0'].max()
            if local_max > global_max:
                global_max = local_max
            examples.append(a)
        return examples,labels, global_max+1
    finally:
        f.close()
