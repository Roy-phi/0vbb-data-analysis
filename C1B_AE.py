#!/usr/bin/env python
# coding: utf-8

# In[ ]:


env='server'
#env='local'
if(env=='server'):
    packpath=r'/share/003/users/zhangbt/C1B_new/RootFile/maketree/pypack'
    filename="/share/002/rawdata/C1B/20210512/20210512_1kg_pre4000_Ihb10ms_Tg200us_Th11_6us1.2x50_12us0.5x5_out10x5_Tout10x1.5_RT20s_Th228FADC_RAW_Data_%d.bin"
    savename="./AE_par/"
    mfilter_path='/share/003/users/zhangbt/C1B_new/RootFile/maketree/pypack/mfilter.npy'
    it_num=4
    file_num=4
elif(env=='local'):
    packpath=r'E:\jupyter-notebook\packages'
    filename="G:/CDEX/DATA/20200331_t3_Th_BEGe_2026_6us_0.5_5_2111_0.5_10_Difout10_FADC_RAW_Data_%d.bin"#modify
    savename="G:/CDEX/wave_par/"
    mfilter_path=r"E:/CDEX/0vbb/Code/AE_par/mfilter.npy"
    it_num=4
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
    np.save(result_name+"A_625",A)
    
    E=result[0][1]
    for i in range(1,len(result)):
        E=np.append(E,result[i][1],axis=0)
    np.save(result_name+"E_625",E)

if __name__=="__main__":
    
    pool = multiprocessing.Pool()
    it=waveform.get_it(filename,range(file_num),it_num,mfilter=mfilter_path)
    result = pool.starmap_async(waveform.get_par, it).get()
    pool.close()
    pool.join()
    pack_result(savename)
print("time cost: ")
print(time.time()-time_start," s")   

