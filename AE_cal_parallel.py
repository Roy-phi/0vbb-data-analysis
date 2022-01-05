#!/usr/bin/env python
# coding: utf-8

# In[3]:


env='server'
#env='local'
if(env=='server'):
    packpath=r'/home/zhangbt/pycode/mypackage'
    filename="/lustre/BEGe_BNU/20200331/20200331_t3_Th_BEGe_2026_6us_0.5_5_2111_0.5_10_Difout10_FADC_RAW_Data_%d.bin"
    savename="./AE_par/"
    mfilter_path='./AE_par/mfilter.npy'
    it_num=32
    file_num=188
elif(env=='local'):
    packpath=r'E:\jupyter-notebook\packages'
    filename="G:/CDEX/DATA/20200331_t3_Th_BEGe_2026_6us_0.5_5_2111_0.5_10_Difout10_FADC_RAW_Data_%d.bin"#modify
    savename="G:/CDEX/wave_par/"
    mfilter_path=r"E:/CDEX/0vbb/Code/AE_par/mfilter.npy"
    it_num=2
    file_num=4

import sys
sys.path.append(packpath)# modify
import time
import multiprocessing
import numpy as np
import waveform

time_start=time.time()


def pack_result(result_name):
    A=result[0][0]
    for i in range(1,len(result)):
        A=np.append(A,result[i][0],axis=0)
    np.save(result_name+"A_bio_6_8",A)
    
    E=result[0][1]
    for i in range(1,len(result)):
        E=np.append(E,result[i][1],axis=0)
    np.save(result_name+"E_trap",E)

if __name__=="__main__":
    
    pool = multiprocessing.Pool()
    it=waveform.get_it(filename,range(file_num),it_num,mfilter=mfilter_path)
    result = pool.starmap_async(waveform.get_par, it).get()
    pool.close()
    pool.join()
    pack_result(savename)
print("time cost: ")
print(time.time()-time_start," s")   

