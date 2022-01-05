#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[4]:


#### Wavelet denoising
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.interpolate import interp1d
import time as ti
from scipy.fftpack import fft,ifft
from scipy.fft import fftshift

def get_par(filename,it,mfilter_=None):
    wave=waveform(filename,it,mfilter=mfilter_)
    return wave.iterate()

def get_it(filename,iterator,it_num,mfilter=None):
    length=int(np.ceil(len(iterator)/it_num))
    it=[]
    for i in range(int(np.ceil(len(iterator)/length))):
        a=(filename,range(i*length,min((i+1)*length,len(iterator))),mfilter)
        it.append(a)
    return tuple(it)

class waveform(object):
    def __init__(self,filename,iterator,par_num={"A":2,"E":4},window=12000,eventN=10000,mfilter=None,Taos=10,FT=2,L=14):
        self.filename=filename
        self.iterator=iterator
        self.window=window
        self.raww=np.zeros((window,))
        self.processedw=np.zeros((window,))
        self.opch=8 #open channel number
        self.eventN=eventN
        self.maxindex=0
        self.max=0
        self.rise_point=0
        self.mfilter=None
        if(mfilter!=None):
            self.mfilter=np.load(mfilter)       
        
        #creating wavelet and define parameters
        
        #### ZAC filter:
        us=1
#         Taos=10*us              #cusp parameter            #
#         FT=1*us                 #flat top width
#         L=16*us                 #length of one cusp side
        Tao=100*us              #pre amp time constant  
        deltaT=0.01*us            #sampling interval
        self.FF,self.ZAC=self.ZACfilter(Taos,L,FT,deltaT,Tao)

        self.A=np.zeros((len(iterator)*eventN,par_num["A"]))
        self.E=np.zeros((len(iterator)*eventN,par_num["E"]))
    
    def read_event(self,filename,event_index,channel=1):#channel=1 means preamplifier

        if(channel>=self.opch):
            print("channel number error")
            return
        run = int(event_index/self.eventN)
        evt = event_index%self.eventN+1

        stream = open(filename,"rb")

        stream.seek(168,0)                                           #skip the header, 168 is file header size
        stream.seek((4*9+2*self.opch*self.window)*(evt-1)+4*9,1) #skip to the target event,and skip the event header info(size is 4*9) 

        self.raww=np.fromfile(stream,dtype="H",count= self.window,offset= self.window*2*channel).astype(int)       
        
    def iterate(self,iterator_=None,path="G:/CDEX/wave_par/%d_%d",Aname="A",Ename="E"):
        count=-1
        ##########################if want to use multi thread, should give iterator
        it=[]
        if(iterator_==None):
            it=[n for n in self.iterator]
        if(iterator_!=None):
            it=[n for n in iterator_]

        ##########################read and processe the wave
        #print(it)
        for n in it:
            filename=self.filename%n
            for j in range(self.eventN):   
                count+=1                 ###############this line required the iterator is [...,1,2,...N]
                if(count%1000==0):
                    print(j+n*self.eventN)
                ##############filtered inhibit <4000
                self.read_event(filename,j,channel=5)
                if(np.max(self.raww)>4000):
                    continue
                ############# read pulse:
                self.read_event(filename,j,channel=2)
                self.rise_point=0
            
# ######################## get E
                wave_length=4096
                threshold=0.5
### trapzoid
                En=0
                self.cut_around_risep(threshold,wave_length,Invert=1)
                L=2
                T=400
                self.TrapShapping(L,L+T,2*L+T,-1e-4)
                self.MWA(T-100)
                self.E[count,En]=self.max_amp()
                En+=1
### zac
                self.cut_around_risep(threshold,wave_length,Invert=1)              
                self.E[count,En]=np.max(self.ZACshapping(self.processedw,self.FF))
                En+=1
            
### partial and MWA
                self.cut_around_risep(threshold,wave_length,Invert=1)
                self.Partial(Npoint=600)
                self.NMWA(Npoint=600,Ntimes=2)
                self.E[count,En]=self.max_amp()
                En+=1
                
### max amp
                self.cut_around_risep(threshold,wave_length,Invert=1)
                self.E[count,En]=self.max_amp()
                En+=1

# ### main amplifier max
#                 self.read_event(filename,j,channel=0)
#                 self.E[count,En]=np.max(self.raww)
#                 En+=1
                
# ####get E done   

#                 self.read_event(filename,j,channel=1)  

# ######################### get A 
                k=0         ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #GERDA method
#                 #####
#                 MAWindow=21
#                 inter_num=5
#                 inter_win=2048
#                 threshold=np.linspace(2,5,2,endpoint=True)
                
#                 #plt.plot(self.raww)
#                 self.cut_around_risep(0.5,inter_win,MWApn=21)

# #                 plt.plot(self.processedw)
# #                 self.rise_p()
# #                 print(self.rise_point-int(MAWindow/2))
# #                 self.cut_around(0.5,win=inter_win,point=self.rise_point-int(MAWindow/2))  ##phase lock

#                 self.interpolate(inter_num,'cubic')
#                 pulse_interpolated=np.zeros((inter_win*inter_num,))
#                 pulse_interpolated[:]=self.processedw[:]                            ##save temp pulse,used after

#                 self.Partial(inter_num*5)
#                 pulse_partialed=np.zeros((inter_win*inter_num,))
#                 pulse_partialed[:]=self.processedw[:]                               ##save temp pulse,used after

                
#                 for thre in range(len(threshold)):
#                     self.processedw[:]=pulse_partialed[:]
#                     self.wavelet_denoise('bior6.8',thre)

#                     self.A[count,k]=self.max_amp()

#                     k+=1
#                 ###########################################
                
# #CDEX method 
#                 ####
#                 for thre in range(len(threshold)):
#                     self.processedw[:]=pulse_interpolated[:]
#                     self.wavelet_denoise('bior6.8',thre)
#                     self.Partial(inter_num*5)
#                     self.A[count,k]=self.max_amp()
#                     k+=1
#                 ###########################################

#MWA method   
                ###
                self.cut_around_risep(0.5,4096,Invert=1)
                self.NMWA(5,3)                ##5point 3times moving window average
                #self.interpolate(inter_num,'cubic')
                self.Partial(3)
                self.A[count,k]=self.max_amp()
                k+=1
                ###########################################
                
                self.cut_around_risep(0.5,4096,Invert=1)
                self.NMWA(5,3)                ##5point 1times moving window average
                #self.interpolate(inter_num,'cubic')
                self.Partial(5)
                self.A[count,k]=self.max_amp()
                k+=1
                
# #Match filter method
#                 ###
#                 if(type(self.mfilter) is np.ndarray):
#                     L=len(self.mfilter)
#                     self.cut_around_risep(0.5,L)
#                     self.Partial(5)
#                     ####fft and ifft
#                     Out=fftshift(ifft(fft(self.processedw)*self.mfilter))
#                     self.A[count,k]=np.max(abs(Out))
#                     k+=1

# #Fast amplifier max
#                 ###
#                 self.read_event(filename,j,channel=2)
#                 self.cut_around_risep(threshold=0.5,win=2500,Invert=1)
#                 self.A[count,k]=self.max_amp(Length=800)
#                 k+=1
                
#                 self.interpolate(inter_num,'cubic')   ##interpolate 5 point
#                 self.A[count,k]=self.max_amp(Length=4000)
#                 k+=1
                
#get A done
        return self.A,self.E
                
                

#                 ###
#                 threshold=[1,2,3,4,5]
#                 k=0
#                 for i in range(5):
#                     self.cut_around_risep(0.5,4096)
#                     self.wavelet_denoise('db5',3)
#                     self.Partial(5)
#                     self.A[count,k]=self.max_amp()
#                     k+=1
#                 ###
#                 win=2048
#                 self.cut_around_risep(0.5,win)
#                 self.wavelet_denoise('db5',3)
#                 self.interpolate(5,'cubic')
#                 self.Partial(25)
#                 self.A[count,k]=self.max_amp()
#                 k+=1
                #######

                
    def save(self,path="G:/CDEX/wave_par/",Aname=None,Ename=None):
        if(Aname!=None):
            np.save(path+Aname,self.A)
        if(Ename!=None):
            np.save(path+Ename,self.E)          
            

    
    def AssignRange(self,start=0,end=12000):
        self.processedw=np.zeros((end-start,))
        self.processedw[:]=self.raww[start:end]

    def Partial(self,Npoint):
        py=np.array([1])
        px=np.zeros((Npoint+1,))
        px[0]=1
        px[Npoint]=-1
        self.processedw=signal.lfilter(px,py,self.processedw)
        
    
    def MWA(self,Npoint):
        py=np.array([1])
        px=np.zeros((Npoint,))+1./Npoint
        self.processedw=signal.lfilter(px,py,self.processedw)


    def NMWA(self,Npoint,Ntimes):
        for i in range(Ntimes):
            self.MWA(Npoint)
    

        
    def invert(self,Length=1000,par=-1):
        self.processedw=par*(self.processedw-self.processedw[0:Length].mean())
    
    def max_amp(self,Length=1000):
        return self.processedw.max()-self.processedw[0:Length].mean()
    
    def rise_p(self,threshold=0.5):
        self.maxindex=self.processedw.argmax()
        for i in range(self.maxindex,0,-1):
            if(self.processedw[i]<=threshold*self.processedw[self.maxindex]):
                self.rise_point=i
                return i

    def cut_around_risep(self,threshold=0.5,win=4096,Invert=-1,MWApn=None):
        self.AssignRange()

        self.invert(par=Invert)

        if(MWApn!=None):
            self.MWA(MWApn)
            self.rise_p(threshold)
            self.rise_point=self.rise_point-int(MWApn/2)
        else:
            self.rise_p(threshold)
        self.rise_point=min(self.rise_point,self.window-win/2)
        self.rise_point=max(self.rise_point,win/2)
        L=int(self.rise_point-win/2)
        H=int(self.rise_point+win/2)
        self.processedw=self.processedw[L:H]
        
    def cut_around(self,threshold=0.5,win=4096,point=3000,Invert=-1):
        self.AssignRange()

        self.invert(par=Invert)
        
        point=min(point,self.window-win/2)
        point=max(point,win/2)
        L=int(point-win/2)
        H=int(point+win/2)
        self.processedw=self.processedw[L:H]
        
        
    def TrapShapping(self,na,nb,nc,theta):
        a=np.array([1,-2,1])
        b=np.zeros((nc+3,))
        e=math.exp(theta)
        b[1]=1./na
        b[na+1]=-1./na
        b[nb+1]=-1./na
        b[nc+1]=1./na
        b[2]=-e/na
        b[na+2]=e/na
        b[nb+2]=e/na
        b[nc+2]=-e/na
        self.processedw=signal.lfilter(b,a,self.processedw)

    def wavelet_denoise(self,wave,threshold):
        wavelet=pywt.Wavelet(wave)
        maxlev=pywt.dwt_max_level(len(self.processedw),wavelet)
        coeffs=pywt.wavedec(self.processedw,wavelet,level=maxlev) #将信号进行小波分解
        coeffs=sigmoid_th(coeffs,threshold,3)
        self.processedw=pywt.waverec(coeffs,wave)
        
    def interpolate(self,in_num,in_kind='cubic'):
        length=len(self.processedw)
        x_t=np.linspace(0,length-1,length,endpoint=True)
        f=interp1d(x_t,self.processedw,kind=in_kind)
        
        new_x=np.linspace(0,length-1,length*in_num,endpoint=True)
        self.processedw=f(new_x)

    def read_header(self):
        run_filename=self.filename%self.iterator[0]
        stream = open(run_filename,"rb")
        print("open file ",run_filename)
        pstt = 0.
        FiredD = 0
        V1724_1_DAC = [0]*8
        V1724_1_Tg = [0]*8
        V1724_1_twd = 0
        V1724_1_pretg = 0
        V1724_1_opch = 0
        V1724_2_DAC = [0]*8
        V1724_2_Tg = [0]*8
        V1724_2_twd = 0
        V1724_2_pretg = 0
        V1724_2_opch = 0
        V1721_DAC = [0]*8 #V1724-1 Channel DAC     
        V1721_Tg = [0]*8  #V1724-1 Trigger Settings
        V1721_twd = 0      #V1724-1 Time Window     
        V1721_pretg = 0    #V1724-1 Pre Trigger     
        V1721_opch = 0     #V1724-1 Opened Channel
        V1729_th_DAC = 0
        V1729_posttg = 0
        V1729_tgtype = 0
        V1729_opch = 0
        rstt = 0.
        redt = 0.

        print("***************************Run Header**************************************\n");
        pstt = struct.unpack("d", stream.read(8))[0]
        print("* Program Start Time:",pstt," s.\n");
        FiredD = struct.unpack("I", stream.read(4))[0]
        print("* Fired Devices: ",FiredD," ( V1724-1 | V1724-2 | V1729)\n\n")

        print("* V1724-1 Channel DAC:      ")
        for i in range(8):
            V1724_1_DAC[i] = struct.unpack("I", stream.read(4))[0]
            print(V1724_1_DAC[i],"\t")
        print("\n")

        V1724_1_twd = struct.unpack("I", stream.read(4))[0]
        print("* V1724-1 Time Window:  ",V1724_1_twd,"\n")
        V1724_1_pretg = struct.unpack("I", stream.read(4))[0]
        print("* V1724-1 Pre Trigger: ",V1724_1_pretg,"\n")
        V1724_1_opch = struct.unpack("I", stream.read(4))[0]
        print("* V1724-1 Opened Channel: ",V1724_1_opch,"\n\n")

        #V1724-2 Settings
        print("* V1724-2 Channel DAC:      ");
        for i in range(8):
            V1724_2_DAC[i] = struct.unpack("I", stream.read(4))[0]
            print(V1724_2_DAC[i],"\t")
        print("\n")

        V1724_2_twd = struct.unpack("I", stream.read(4))[0]
        print("* V1724-2 Time Window:  ",V1724_2_twd,"\n")
        V1724_2_pretg = struct.unpack("I", stream.read(4))[0]
        print("* V1724-2 Pre Trigger: ",V1724_2_pretg,"\n")
        V1724_2_opch = struct.unpack("I", stream.read(4))[0]
        print("* V1724-2 Opened Channel: ",V1724_2_opch,"\n\n")

        V1729_th_DAC = struct.unpack("I", stream.read(4))[0]
        print("* V1729 Threshold DAC:  ",V1729_th_DAC,"\n")
        V1729_posttg = struct.unpack("I", stream.read(4))[0]
        print("* V1729 Post Trigger: ",V1729_posttg,"\n")
        V1729_tgtype = struct.unpack("I", stream.read(4))[0]
        print("* V1729 Trigger Type: ",V1729_tgtype,"\n")
        V1729_opch = struct.unpack("I", stream.read(4))[0]
        print("* V1729 Opened Channel: ",V1729_opch,"\n\n")

        #V1721 Settings
        print("* V1721 Channel DAC:        ")
        for i in range(8):
            V1724_2_DAC[i] = struct.unpack("I", stream.read(4))[0]
            print(V1724_2_DAC[i],"\t")
        print("\n")
        V1721_twd = struct.unpack("I", stream.read(4))[0]
        print("* V1721 Time Window: ",V1721_twd,"\n")
        V1721_pretg = struct.unpack("I", stream.read(4))[0]
        print("* V1721 Pre Trigger: ",V1721_pretg,"\n")
        V1721_opch = struct.unpack("I", stream.read(4))[0]
        print("* V1721 Opened Channel: ",V1724_1_opch,"\n")

        rstt = struct.unpack("d", stream.read(8))[0]
        print("* Run Start Time: ",rstt," s.\n")
        print("***************************************************************************\n");

        #event header
        Hit_pat = 0
        V1729_tg_rec = 0
        Evt_deadtime = 0
        Evt_starttime = 0
        Evt_endtime = 0
        V1724_1_tgno = 0
        V1724_2_tgno = 0
        V1721_tgno = 0
        V1724_1_tag = 0
        print("current point position: ",stream.tell(),"\n")
    #     Hit_pat = struct.unpack("I", stream.read(4))[0]
    #     V1729_tg_rec = struct.unpack("I", stream.read(4))[0]
    #     Evt_deadtime = struct.unpack("I", stream.read(4))[0]
    #     Evt_starttime = struct.unpack("I", stream.read(4))[0]
    #     Evt_endtime = struct.unpack("I", stream.read(4))[0]
    #     V1724_1_tgno = struct.unpack("I", stream.read(4))[0]
    #     V1724_2_tgno = struct.unpack("I", stream.read(4))[0]
    #     V1721_tgno = struct.unpack("I", stream.read(4))[0]
    #     V1724_1_tag = struct.unpack("I", stream.read(4))[0]
        self.window=V1724_1_twd
        self.opch=V1724_1_opch
        return V1724_1_opch,V1724_1_twd

    def ZACfilter(self,Taos,L,FT,deltaT,Tao):
        A=3*(2*Taos*(np.cosh(L/Taos)-1)+FT*np.sinh(L/Taos))/L**3
        Filter_length=2*L+FT
        Npoint=int(Filter_length/deltaT)
        conv=-np.exp(-deltaT/Tao)
        ZAC=np.arange(Npoint,dtype=float)
        FF =np.arange(Npoint-1,dtype=float)
        ####################################################################
        for i in np.arange(Npoint):
            t=deltaT*i
            if(0<t and t<L ):
                ZAC[i]=np.sinh(t/Taos)+A*((t-L/2)**2-(L/2)**2)
            elif(L<=t and t<L+FT):
                ZAC[i]=np.sinh(L/Taos)
            elif(L+FT<=t and t<2*L+FT):
                ZAC[i]=np.sinh((2*L+FT-t)/Taos)+A*((3/2*L+FT-t)**2-(L/2)**2)
        #####################################################################
        #plt.plot(ZAC)
        for i in np.arange(Npoint-1):
            FF[i]=ZAC[i]*conv+ZAC[i+1]
        #####################################################################
        return FF,ZAC

    def ZACshapping(self,Vin,FF):
        N_zac=FF.shape[0]
        N_vin=Vin.shape[0]
        N_vout=N_vin-N_zac+2
        Vout=signal.fftconvolve(FF,Vin)
        return Vout[N_zac:N_vin]
        
def Gerda_sigmoid_th(coeffs,Th=1,s=0.2):
    for i in range(len(coeffs)):
        coeffs[i]=coeffs[i]/(1+np.exp(1/(s*(1-abs(coeffs[i])/Th))))
    return coeffs     

def sigmoid_th(coeffs,Th=2,s=3):
    for i in range(len(coeffs)):
        coeffs[i]=coeffs[i]*np.maximum(0,-1+2/(1+np.exp(-((abs(coeffs[i])-Th)*s))))
    return coeffs

