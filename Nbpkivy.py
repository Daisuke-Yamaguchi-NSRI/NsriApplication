import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import cmath
import seaborn as sns
import pandas as pd
import sympy
import sys
import NsriMain_data
mpl.use('TkAgg')

from matplotlib.animation import ArtistAnimation
from scipy.stats import multivariate_normal
from scipy.special import hyp2f1
from scipy.special import gamma
from scipy.special import spherical_jn
from scipy.special import sph_harm
from scipy.special import poch
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

#main function
def NbpMain():
	fig=plt.figure(figsize=(6,6))
	ax=fig.add_subplot(111)
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	ims=[]

	frames=[]

	X1=[0]*20000
	X2=[0]*20000
	X3=[0]*20000

	X1_all=[]
	X2_all=[]
	X3_all=[]

	Y1=[0]*20000
	Y2=[0]*20000
	Y3=[0]*20000

	Y1_all=[]
	Y2_all=[]
	Y3_all=[]


	X1d=[0]*20000
	X2d=[0]*20000
	X3d=[0]*20000

	Y1d=[0]*20000
	Y2d=[0]*20000
	Y3d=[0]*20000

	#initial condition
	dt=0.005

	X1[0]=float(NsriMain_data.X10)
	Y1[0]=float(NsriMain_data.Y10)

	X2[0]=float(NsriMain_data.X20)
	Y2[0]=float(NsriMain_data.Y20)

	X3[0]=float(NsriMain_data.X30)
	Y3[0]=float(NsriMain_data.Y30)

	X1d[0]=float(NsriMain_data.X1d0)
	Y1d[0]=float(NsriMain_data.Y1d0)

	X2d[0]=float(NsriMain_data.X2d0)
	Y2d[0]=float(NsriMain_data.Y2d0)

	X3d[0]=float(NsriMain_data.X3d0)
	Y3d[0]=float(NsriMain_data.Y3d0)



	for i in np.arange(0,2000,1):
		#3体目の位置と速度
		X3[i+1]=X3[i]+X3d[i]*dt
		Y3[i+1]=Y3[i]+Y3d[i]*dt

		X3d[i+1]=X3d[i]+(-(1/(np.power(X3[i]-X1[i],2)+np.power(Y3[i]-Y1[i],2)))*((X3[i]-X1[i])/np.sqrt(np.power(X3[i]-X1[i],2)+np.power(Y3[i]-Y1[i],2)))-(1/(np.power(X3[i]-X2[i],2)+np.power(Y3[i]-Y2[i],2)))*((X3[i]-X2[i])/np.sqrt(np.power(X3[i]-X2[i],2)+np.power(Y3[i]-Y2[i],2))))*dt
		Y3d[i+1]=Y3d[i]+(-(1/(np.power(X3[i]-X1[i],2)+np.power(Y3[i]-Y1[i],2)))*((Y3[i]-Y1[i])/np.sqrt(np.power(X3[i]-X1[i],2)+np.power(Y3[i]-Y1[i],2)))-(1/(np.power(X3[i]-X2[i],2)+np.power(Y3[i]-Y2[i],2)))*((Y3[i]-Y2[i])/np.sqrt(np.power(X3[i]-X2[i],2)+np.power(Y3[i]-Y2[i],2))))*dt;


		#2体目の位置と速度
		X2[i+1]=X2[i]+X2d[i]*dt
		Y2[i+1]=Y2[i]+Y2d[i]*dt

		X2d[i+1]=X2d[i]+(-(1/(np.power(X2[i]-X1[i],2)+np.power(Y2[i]-Y1[i],2)))*((X2[i]-X1[i])/np.sqrt(np.power(X2[i]-X1[i],2)+np.power(Y2[i]-Y1[i],2)))-(1/(np.power(X2[i]-X3[i],2)+np.power(Y2[i]-Y3[i],2)))*((X2[i]-X3[i])/np.sqrt(np.power(X2[i]-X3[i],2)+np.power(Y2[i]-Y3[i],2))))*dt
		Y2d[i+1]=Y2d[i]+(-(1/(np.power(X2[i]-X1[i],2)+np.power(Y2[i]-Y1[i],2)))*((Y2[i]-Y1[i])/np.sqrt(np.power(X2[i]-X1[i],2)+np.power(Y2[i]-Y1[i],2)))-(1/(np.power(X2[i]-X3[i],2)+np.power(Y2[i]-Y3[i],2)))*((Y2[i]-Y3[i])/np.sqrt(np.power(X2[i]-X3[i],2)+np.power(Y2[i]-Y3[i],2))))*dt

		#1体目の位置と速度

		X1[i+1]=X1[i]+X1d[i]*dt
		Y1[i+1]=Y1[i]+Y1d[i]*dt

		X1d[i+1]=X1d[i]+(-(1/(np.power(X1[i]-X2[i],2)+np.power(Y1[i]-Y2[i],2)))*((X1[i]-X2[i])/np.sqrt(np.power(X1[i]-X2[i],2)+np.power(Y1[i]-Y2[i],2)))-(1/(np.power(X1[i]-X3[i],2)+np.power(Y1[i]-Y3[i],2)))*((X1[i]-X3[i])/np.sqrt(np.power(X1[i]-X3[i],2)+np.power(Y1[i]-Y3[i],2))))*dt
		Y1d[i+1]=Y1d[i]+(-(1/(np.power(X1[i]-X2[i],2)+np.power(Y1[i]-Y2[i],2)))*((Y1[i]-Y2[i])/np.sqrt(np.power(X1[i]-X2[i],2)+np.power(Y1[i]-Y2[i],2)))-(1/(np.power(X1[i]-X3[i],2)+np.power(Y1[i]-Y3[i],2)))*((Y1[i]-Y3[i])/np.sqrt(np.power(X1[i]-X3[i],2)+np.power(Y1[i]-Y3[i],2))))*dt
	
		X1_all.append(X1[i])
		X2_all.append(X2[i])
		X3_all.append(X3[i])
		Y1_all.append(Y1[i])
		Y2_all.append(Y2[i])
		Y3_all.append(Y3[i])


		im1=ax.plot(X1[i],Y1[i],"o",X1_all,Y1_all,'--',color="blue")
		im2=ax.plot(X2[i],Y2[i],"o",X2_all,Y2_all,'--',color="red")
		im3=ax.plot(X3[i],Y3[i],"o",X3_all,Y3_all,'--',color="green")
		ims.append(im1+im2+im3)

	anim=ArtistAnimation(fig,ims,interval=100,repeat="True",repeat_delay=4000)
	anim.save('3bodyproblem.gif',writer="pillow")
	plt.show()




