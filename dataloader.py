import os
import numpy as np
import csv
#（samples，input_dim1，input_dim2, input_dim3，channels）
def data_loader(x,y,z,mode):
    if mode=='train':
        listing=os.listdir('train_val')
        dataset=np.empty((3*len(listing),x,y,z))
        n=0
        for addr in listing:
            cdd='train_val/'+addr
            data=np.load(cdd)
            vox=data['voxel'][50-z//2:50+z//2]
            seg=data['seg'][50-z//2:50+z//2]
            vox_cut=np.zeros((x,y,z))
            vox_fliplr=np.zeros((x,y,z))
            vox_flipud=np.zeros((x,y,z))
            seg_cut=np.zeros((x,y,z))
            for k in range(z):
                vox_cut[k]=vox[k][50-x//2:50+x//2,50-y//2:50+y//2]
                seg_cut[k]=seg[k][50-x//2:50+x//2,50-y//2:50+y//2]    #get datablock of x*y*z
                vox_cut[k]=np.multiply(vox_cut[k],seg_cut[k])      #apply mask
                vox_fliplr[k]=np.fliplr(vox_cut[k])
                vox_flipud[k]=np.flipud(vox_cut[k])
            dataset[3*n,:,:,:]=vox_cut.astype('float32')/255
            dataset[3*n+1,:,:,:]=vox_fliplr.astype('float32')/255
            dataset[3*n+2,:,:,:]=vox_flipud.astype('float32')/255
            n=n+1
        dataset=dataset.reshape(3*n,x,y,z,1)
        print(mode+'set loading done')
        return dataset
    elif mode=='test':
        listing=os.listdir('test')
        dataset=np.empty((len(listing),x,y,z))
        n=0
        for addr in listing:
            cdd='test/'+addr
            data=np.load(cdd)
            vox=data['voxel'][50-z//2:50+z//2]
            seg=data['seg'][50-z//2:50+z//2]
            vox_cut=np.zeros((x,y,z))
            seg_cut=np.zeros((x,y,z))
            for k in range(z):
                vox_cut[k]=vox[k][50-x//2:50+x//2,50-y//2:50+y//2]
                seg_cut[k]=seg[k][50-x//2:50+x//2,50-y//2:50+y//2]    #get datablock of x*y*z
                vox_cut[k]=np.multiply(vox_cut[k],seg_cut[k])      #apply mask
            dataset[n,:,:,:]=vox_cut.astype('float32')/255
            n=n+1
        dataset=dataset.reshape(n,x,y,z,1)
        print(mode+'set loading done')
        return dataset
    else:
        with open('train_val.csv')as t:
            t_csv=csv.reader(t)
            tmp=np.array([])
            for row in t_csv:
                tmp=np.append(tmp,row[1])
                tmp=np.append(tmp,row[1])
                tmp=np.append(tmp,row[1])
            return tmp
    
    
    #print(dataset.size)