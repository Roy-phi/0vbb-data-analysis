import numpy as np
import keras
import math
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Bidirectional, Concatenate, Input, Dense, Dropout, Flatten, Multiply, Add
from keras.layers import Conv2D, MaxPooling2D, GRU, Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.layers.core import Lambda

from PIL import Image, ImageFile
import os
import cv2
from scipy.fftpack import fft

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#参数配置
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ImageFile.LOAD_TRUNCATED_IMAGES = True

#建模
def build_model():

	gru_input_size = 32
	att_size = 10

	#============ 输入数据形状 =============
	LI0 = Input(shape = (224+64,224,3))

	#============ 构建频域模型 =============
	#从输入数据中分离出给频域的部分
	LFFT1 = Lambda(lambda x: x[:,224:224+64,:125,0])(LI0)
	LFFT2 = Lambda(lambda x: x[:,224:224+64,:125,1])(LI0)
	LFFT = Concatenate(axis = 2)([LFFT1,LFFT2])
	#构建三层1D卷积网络
	FF1 = Conv1D(32,3,activation='relu')
	FM1 = MaxPooling1D(pool_size=2)
	FF2 = Conv1D(64,3,activation='relu')
	FM2 = MaxPooling1D(pool_size=2)
	FF3 = Conv1D(128,3,activation='relu')
	FM3 = MaxPooling1D(pool_size=2)
	#64个向量分别输入三层卷积网络和一层全连接
	OF = []
	for i in range(64):
		Hi_raw = Lambda(lambda x:x[:,i,:])(LFFT)
		Hi = Lambda(lambda x:K.expand_dims(x,axis=-1))(Hi_raw)
		Oi_raw = FM3(FF3(FM2(FF2(FM1(FF1(Hi))))))
		Oi_f = Flatten()(Oi_raw)
		Oi = Dense(1,activation='linear')(Oi_f)
		OF.append(Oi)
	COF = Concatenate(axis = -1)(OF) #频域输出的w向量（64位）
	#============ 频域模型构建完毕 ============

	#============ 构建像素域模型 ============
	#从输入数据中分离出像素域的部分
	LI = Lambda(lambda x:x[:,:224,:,:])(LI0) #LI = Input(shape = (224, 224, 3))
	#第一层卷积
	LC11 = Conv2D(32, (3, 3), activation='relu')(LI)
	LC12 = Conv2D(32, (1, 1), activation='relu')(LC11)
	LC13 = MaxPooling2D(pool_size=(2,2))(LC12)
	#Branch1 
	LB11 = Conv2D(64, (1, 1), activation='relu')(LC13)
	LB12 = Flatten()(LB11)
	v1 = Dense(gru_input_size,activation='linear')(LB12)
	v1 = Dropout(0.5)(v1)
	#第二层卷积
	LC21 = Conv2D(64, (3, 3), activation='relu')(LC13)
	LC22 = Conv2D(64, (1, 1), activation='relu')(LC21)
	LC23 = MaxPooling2D(pool_size=(2,2))(LC22)
	#Branch2
	LB21 = Conv2D(64, (1, 1), activation='relu')(LC23)
	LB22 = Flatten()(LB21)
	v2 = Dense(gru_input_size,activation='linear')(LB22)
	v2 = Dropout(0.5)(v2)
	#第三层卷积
	LC31 = Conv2D(64, (3, 3), activation='relu')(LC23)
	LC32 = Conv2D(64, (1, 1), activation='relu')(LC31)
	LC33 = MaxPooling2D(pool_size=(2,2))(LC32)
	#Branch3
	LB31 = Conv2D(64, (1, 1), activation='relu')(LC33)
	LB32 = Flatten()(LB31)
	v3 = Dense(gru_input_size,activation='linear')(LB32)
	v3 = Dropout(0.5)(v3)
	#第四层卷积
	LC41 = Conv2D(128, (3, 3), activation='relu')(LC33)
	LC42 = Conv2D(128, (1, 1), activation='relu')(LC41)
	LC43 = MaxPooling2D(pool_size=(2,2))(LC42)
	#Branch4
	LB41 = Conv2D(64, (1, 1), activation='relu')(LC43)
	LB42 = Flatten()(LB41)
	v4 = Dense(gru_input_size,activation='linear')(LB42)
	v4 = Dropout(0.5)(v4)
	#添加一个维度，将张量v1,v2,v3,v4的形状从(None,32)变为(None,1,32)
	v1 = Lambda(lambda x:K.expand_dims(x,axis=1))(v1) #(None,1,32)
	v2 = Lambda(lambda x:K.expand_dims(x,axis=1))(v2)
	v3 = Lambda(lambda x:K.expand_dims(x,axis=1))(v3)
	v4 = Lambda(lambda x:K.expand_dims(x,axis=1))(v4)
	#将v1,v2,v3,v4合并为(None,4,32)的张量
	M = Concatenate(axis=1)([v1, v2, v3, v4]) #merge([v1, v2, v3, v4], mode='concat', concat_axis=0) #M = Lambda(lambda x:K.expand_dims(x,axis=-1))(M)
	#4x32的张量输入双向GRU，得到4个输出，每个输出为lt=[lx,ly]为32+32=64位
	G = Bidirectional(GRU(gru_input_size, return_sequences=True, input_shape = (4,gru_input_size)), merge_mode = 'concat')(M)
	#============ 像素域模型构建完毕 ============

	#============ 构建Attention ==========
	#得到l0，l1，l2，l3，l4，其中l0来自频域，化为与其他张量相同的64位
	G0 = Dense(gru_input_size*2, activation='linear')(COF) #（None,64)
	G1 = Lambda(lambda x:x[:,0,:])(G) # (None,64)
	G2 = Lambda(lambda x:x[:,1,:])(G)
	G3 = Lambda(lambda x:x[:,2,:])(G)
	G4 = Lambda(lambda x:x[:,3,:])(G)
	#计算tanh(Wf*li+bf), 其中Wf*li+bf用一个全连接层实现，tanh用激活函数实现， 输出的维度暂时*盲目的*定为 att_size = 10
	Fun0 = Dense(att_size, activation='tanh')
	tG0 = Fun0(G0) #(None,10)
	tG1 = Fun0(G1)
	tG2 = Fun0(G2)
	tG3 = Fun0(G3)
	tG4 = Fun0(G4)
	#将(None,10)形状的向量变成(None,1,10)形状
	tG0 = Lambda(lambda x:K.expand_dims(x,axis=1))(tG0) #(None,1,10)
	tG1 = Lambda(lambda x:K.expand_dims(x,axis=1))(tG1)
	tG2 = Lambda(lambda x:K.expand_dims(x,axis=1))(tG2)
	tG3 = Lambda(lambda x:K.expand_dims(x,axis=1))(tG3)
	tG4 = Lambda(lambda x:K.expand_dims(x,axis=1))(tG4)
	#计算v^T*__,得到5个标量
	Fun = Dense(1,use_bias=False,activation='linear')
	tG0 = Fun(tG0) #(None,1,1)
	tG1 = Fun(tG1)
	tG2 = Fun(tG2)
	tG3 = Fun(tG3)
	tG4 = Fun(tG4)
	#合并为(None,5,1)的张量
	Gn = Concatenate(axis=1)([tG0, tG1, tG2, tG3, tG4])
	#计算Alpha
	Alpha = Lambda(lambda x: K.softmax(x))(Gn)
	#计算Alpha与l相乘，得到5个ui (None,64)
	#u0
	a0 = Lambda(lambda x: x[:,0,:])(Alpha)
	cnct0 = []
	for i in range(gru_input_size*2):
		cnct0.append(a0)
	a0 = Concatenate(axis=1)(cnct0)
	nG0 = Multiply()([a0,G0])
	#u1
	a1 = Lambda(lambda x: x[:,1,:])(Alpha)
	cnct1 = []
	for i in range(gru_input_size*2):
		cnct1.append(a1)
	a1 = Concatenate(axis=1)(cnct1)
	nG1 = Multiply()([a1,G1])
	#u2
	a2 = Lambda(lambda x: x[:,2,:])(Alpha)
	cnct2 = []
	for i in range(gru_input_size*2):
		cnct2.append(a2)
	a2 = Concatenate(axis=1)(cnct2)
	nG2 = Multiply()([a2,G2])
	#u3
	a3 = Lambda(lambda x: x[:,3,:])(Alpha)
	cnct3 = []
	for i in range(gru_input_size*2):
		cnct3.append(a3)
	a3 = Concatenate(axis=1)(cnct3)
	nG3 = Multiply()([a3,G3])
	#u4
	a4 = Lambda(lambda x: x[:,4,:])(Alpha)
	cnct4 = []
	for i in range(gru_input_size*2):
		cnct4.append(a4)
	a4 = Concatenate(axis=1)(cnct4)
	nG4 = Multiply()([a4,G4])
	#ui相加得到u
	A = Add()([nG0,nG1,nG2,nG3,nG4])
	#u经过一个全链接层得到p
	O = Dense(1,activation='softmax')(A)
	model = Model(inputs = LI0, outputs = O)

	model.summary()

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model

# 数据生成器
class DataGenerator(keras.utils.Sequence):
	def __init__(self, filenames, labels, batch_size):
		self.filenames = filenames
		self.labels = labels
		self.batch_size = batch_size
		self._shuffle()

	def _shuffle(self):
		self.indexes = np.arange(len(self.filenames))
		np.random.shuffle(self.indexes)

	@staticmethod
	def _parse(imgPath):
		xx = np.zeros((224+64,224,3))
		img = Image.open(imgPath)
		out = img.resize((224,224))
		out = np.array(out)
		s = out.shape
		xx[:224,:,:] = out[:,:,:]
		#获取图像频域数据：暂时先*盲目的*只取了通道0做DCT： 先DCT - 再把高度resize成64 - 再做FFT - 最后每行采样250个点
		img = cv2.imread(imgPath)
		img = img[:,:,0]
		if img.shape[0] % 2 == 1:
			img = img[:img.shape[0]-1,:]
		if img.shape[1] % 2 == 1:
			img = img[:,:img.shape[1]-1]
		imgdct = cv2.dct(np.float32(img))
		if imgdct.shape[1] >= 250:
			imgdct = cv2.resize(imgdct, (imgdct.shape[1],64))
		else:
			imgdct = cv2.resize(imgdct, (250,64))
		res = np.zeros((64,250))
		idx =np.linspace(0,imgdct.shape[1]-1,num=250, dtype=int)
		for i in range(64):
			fftres = fft(imgdct[i,:])
			res[i,:] = fftres[idx]
		xx[224:224+64,:125,0] = res[:,:125]
		xx[224:224+64,:125,1] = res[:,125:]
		#end
		return xx

	def on_epoch_end(self):
		self._shuffle()

	def __len__(self):
		return math.ceil(len(self.filenames)/float(self.batch_size))

	def __getitem__(self, idx):
		batch_indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
		batch_x = np.array([self._parse(self.filenames[i]) for i in batch_indexes])
		batch_y = np.array([self.labels[i] for i in batch_indexes])
		return batch_x, batch_y


cnt = 0
filenames = []
labels = []

#读取文件名和labels
def getFiles(folder, y):
	global cnt
	global filenames
	global labels
	root_path = '/home/smy/桌面/DataMining/pic/'+folder+'/'
	dir = root_path
	for root,dir,files in os.walk(dir):
		for file in files:
			fname = str(file)
			if fname[0] == '.':
				continue
			imgPath = root_path+fname
			if imgPath[-4:] != ".gif":
				if os.path.getsize(imgPath) >= 1024*1024*3:
					continue
				img = Image.open(imgPath)
				if img.size[0]*img.size[1] >= 1000*1000:
					continue
				out = img.resize((224,224))
				out = np.array(out)
				s = out.shape
				if len(s) != 3 or s[0] != 224 or s[1] != 224 or s[2] != 3:
					continue
				filenames.append(imgPath)
				labels.append(y)
				cnt += 1
				print(cnt)

getFiles('rumor_pic',0)
getFiles('true_pic_1',1)
getFiles('truth_pic_2',1)
print(len(filenames),len(labels))

#训练前先Shuffle操作一次
L = len(filenames)
idx = np.array(list(range(L)))
np.random.shuffle(idx)
n_filenames = []
n_labels = []
for i in range(len(idx)):
	n_filenames.append(filenames[idx[i]])
	n_labels.append(labels[idx[i]])
filenames = n_filenames
labels = n_labels

#数据生成器，80%训练，20%测试
train_gen = DataGenerator(filenames[:int(0.8*L)], labels[:int(0.8*L)], 32)
test_gen  = DataGenerator(filenames[int(0.8*L):], labels[int(0.8*L):], 32)

#模型
model = build_model()

#训练
model.fit_generator(train_gen, epochs = 1000, validation_data=test_gen, workers = 1, use_multiprocessing=False, shuffle = True)
