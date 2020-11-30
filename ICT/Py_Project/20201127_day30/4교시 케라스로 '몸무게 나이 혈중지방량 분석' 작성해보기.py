#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from mpl_toolkits.mplot3d import Axes3D

#1.0 경우 아래와 같이 쓴다고 한다.
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import RMSprop


# In[2]:


#step1. 데이터 로드& 변수에 할당.
raw_data = np.loadtxt("C:/sourceTree/DataScience/ICT/Py_Project/20201119_day24/Blood_fat.csv",delimiter=",")

xs = np.array(raw_data[:,0], dtype=np.float32)
ys = np.array(raw_data[:,1], dtype=np.float32)
zs = np.array(raw_data[:,2], dtype=np.float32)


# In[3]:


x_data = np.array(raw_data[:,0:2], dtype=np.float32)
y_data = np.array(raw_data[:,2], dtype=np.float32)
y_data = y_data.reshape((25,1))


# In[4]:


rmsprop = RMSprop(lr=0.01)
model = Sequential()
model.add(Dense(1,input_shape=(2,)))
model.compile(loss='mse',optimizer=rmsprop)
model.summary()


# In[5]:


hist = model.fit(x_data,y_data, epochs=1000)
print(hist.history.keys())


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


#딥러닝 구조를 결정합니다(모델을 설정하고 실행하는 부분입니다.)
model = Sequential()
model.add(Dense(1,input_dim=2,activation='relu')) 
model.add(Dense(1,activation='sigmoid'))


# In[7]:


#딥러닝을 실행합니다.
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
#model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
#model.compile(loss='mean_absolute_percentage_error',optimizer='adam',metrics=['accuracy'])
#model.compile(loss='mean_squared_logarthmic_error',optimizer='adam',metrics=['accuracy'])
model.fit(training_data,target_data, epochs=30, batch_size=10)
print(model.evaluate(training_data,model.predict(training_data,target_data), target_data) )


# In[ ]:


#결과를 출력합니다.
print("\n Accuracy: %.4f "%(model.evaluate(training_data,target_data)[1]))


# In[ ]:


#3차원 그래프 그리기
import matplotlib.pyplot as plt #혹시 없는경우 !pip install matplotlib
from mpl_toolkits.mplot3d import Axes3D

x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]
y_data = [y_row[2] for y_row in data]

fig = plt.figure(figsize=(12,12)) #figure(그래프가 그려지는 객체) 생성
ax  = fig.add_subplot(111,projection='3d') #전체공간을 1*1로 잡은 중 첫번째(111), 3d로 표시, 전체공간을 나누는 개념.
                                           #projection='3d' : 표고와 방위각을 지정하여 3d그래프가 보이는 방향을 설정한다.           
#산점도 플롯을 만듦
ax.scatter(x1,x2,y_data) #실제데이터의 그래프를 생성.
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Blood fat')
ax.view_init(15,15)
plt.show()


# In[ ]:


#3차원 그래프 그리기
import matplotlib.pyplot as plt #혹시 없는경우 !pip install matplotlib
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(12,12)) #figure(그래프가 그려지는 객체) 생성
ax  = fig.add_subplot(111,projection='3d') #전체공간을 1*1로 잡은 중 첫번째(111), 3d로 표시, 전체공간을 나누는 개념.
                                           #projection='3d' : 표고와 방위각을 지정하여 3d그래프가 보이는 방향을 설정한다.           
#산점도 플롯을 만듦
ax.scatter(x1,x2,y_data) 
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Blood fat')
ax.view_init(elev=0, azim=0) #보는 각도를 정면으로 설정.
plt.show()


# In[ ]:


#3차원 그래프 그리기
import matplotlib.pyplot as plt #혹시 없는경우 !pip install matplotlib
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(12,12)) #figure(그래프가 그려지는 객체) 생성
ax  = fig.add_subplot(111,projection='3d') #전체공간을 1*1로 잡은 중 첫번째(111), 3d로 표시, 전체공간을 나누는 개념.
                                           #projection='3d' : 표고와 방위각을 지정하여 3d그래프가 보이는 방향을 설정한다.           

#산점도 플롯을 만듦    
ax.scatter(x1,x2,calc_y) #이번에는 예측된 값(calc_y)을 찍는다.
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Blood fat')
ax.view_init(15,15)
plt.show()


# In[ ]:


#3차원 그래프 그리기
import matplotlib.pyplot as plt #혹시 없는경우 !pip install matplotlib
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(12,12)) #figure(그래프가 그려지는 객체) 생성
ax  = fig.add_subplot(111,projection='3d') #전체공간을 1*1로 잡은 중 첫번째(111), 3d로 표시, 전체공간을 나누는 개념.
                                           #projection='3d' : 표고와 방위각을 지정하여 3d그래프가 보이는 방향을 설정한다.           
#산점도 플롯을 만듦
ax.scatter(x1,x2,y_data) #실제데이터의 그래프를 생성.
ax.scatter(x1,x2,calc_y) #이번에는 예측된 값(calc_y)을 찍는다.
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Blood fat')
ax.view_init(15,15)
plt.show()


# In[ ]:


#3차원 그래프 그리기
import matplotlib.pyplot as plt #혹시 없는경우 !pip install matplotlib
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(12,12)) #figure(그래프가 그려지는 객체) 생성
ax  = fig.add_subplot(111,projection='3d') #전체공간을 1*1로 잡은 중 첫번째(111), 3d로 표시, 전체공간을 나누는 개념.
                                           #projection='3d' : 표고와 방위각을 지정하여 3d그래프가 보이는 방향을 설정한다.           
#산점도 플롯을 만듦
ax.scatter(x1,x2,y_data) #실제데이터의 그래프를 생성.
ax.scatter(x1,x2,calc_y) #이번에는 예측된 값(calc_y)을 찍는다.
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Blood fat')
ax.view_init(0,0)
plt.show()

