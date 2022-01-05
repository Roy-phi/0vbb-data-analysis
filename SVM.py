import numpy as np
import math
from PIL import Image, ImageFile
import os
import cv2
from scipy.fftpack import fft
from sklearn import svm

ImageFile.LOAD_TRUNCATED_IMAGES = True

cnt = 0
filenames = []
labels = []

real_pics = set()
wrong_pics = set()

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
					real_pics.add(imgPath)
					filenames.append(imgPath)
					labels.append(y)
					cnt += 1
					print(cnt)
					continue
				img = Image.open(imgPath)
				if img.size[0]*img.size[1] >= 1000*1000:
					real_pics.add(imgPath)
					filenames.append(imgPath)
					labels.append(y)
					cnt += 1
					print(cnt)
					continue
				out = img.resize((224,224))
				out = np.array(out)
				s = out.shape
				if len(s) != 3 or s[0] != 224 or s[1] != 224 or s[2] != 3:
					wrong_pics.add(imgPath)
					filenames.append(imgPath)
					labels.append(y)
					cnt += 1
					print(cnt)
					continue
				filenames.append(imgPath)
				labels.append(y)
				cnt += 1
				print(cnt)

print("Start collecting picture filenames ...")
getFiles('rumor_pic',0)
getFiles('true_pic_1',1)
getFiles('truth_pic_2',1)
print(len(filenames),len(labels))
print("Finished collecting picture filenames.")

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
print("Finished Shuffling.")

#读取图片并生成一个输入数据
def read_data(imgPath):
	ifread = True
	if ifread:
		imgs = cv2.imread(imgPath)
		out = np.zeros((128,128,3))
		for i in range(3):
			img = imgs[:,:,i]

			#====== 选择像素域作为输入 ======
			#imgdct = cv2.resize(img, (128,128))

			#====== 选择离散余弦变换作为输入 ======
			if img.shape[0] % 2 == 1:
				img = img[:img.shape[0]-1,:]
			if img.shape[1] % 2 == 1:
				img = img[:,:img.shape[1]-1]
			imgdct = cv2.resize(cv2.dct(np.float32(img)), (128,128))

			#====== 选择傅立叶变换作为输入 ======
			#imgdct = cv2.resize(np.real(fft(img)), (128,128))

			out[:,:,i] = imgdct
		out = out.flatten()
	return out


#在所有数据中选择大小为count的子集
def chooseData(count,start):
	tot = 0
	jr = [0,0]
	jw = [0,0]
	cx = []
	cy = []
	for i in range(start, len(filenames)):
		if filenames[i] in real_pics:
			jr[labels[i]] += 1
		elif filenames[i] in wrong_pics:
			jw[labels[i]] += 1
		else:
			cx.append(read_data(filenames[i]))
			cy.append(labels[i])
			tot += 1
			if tot % 1000 == 0:
				print(tot)
			if tot == count:
				break
	cx = np.array(cx)
	cy = np.array(cy)
	return cx,cy,jr,jw,i


#获得训练数据
print("Start reading training data ...")
train_x, train_y, train_jr, train_jw, last = chooseData(8000, 0)
print(train_x.shape, train_y.shape)
N_train = train_jr[0]+train_jr[1]+train_jw[0]+train_jw[1]+train_y.shape[0]
print("Scanned Training Sample:",N_train,'( Actual training size:',train_y.shape[0],')')

#训练SVM
print("Start training SVM ...")
model = svm.SVC(C = 10, gamma = 1e-8, kernel='rbf')
model.fit(train_x,train_y)
pred_y = model.predict(train_x)
train_acc = (train_jr[1]+train_jw[0]+np.sum(pred_y==train_y)) / N_train
print("Training Acc:",train_acc)
train_x = None #释放内存
train_y = None

#获得测试数据
print("Start reading testing data ...")
test_x, test_y, test_jr, test_jw, last = chooseData(8000, last+1)
print(test_x.shape, test_y.shape)
N_test = test_jr[0]+test_jr[1]+test_jw[0]+test_jw[1]+test_y.shape[0]
print("Scanned Testing Sample:", N_test, '( Actual testing size:', test_y.shape[0], ')')

#预测结果
print("Start predicting ...")
pred_y = model.predict(test_x)
test_acc = (test_jr[1]+test_jw[0]+np.sum(pred_y == test_y)) / N_test
print("Testing Acc:", test_acc)