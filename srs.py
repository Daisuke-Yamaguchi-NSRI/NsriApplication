import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import cmath
import seaborn as sns
import pandas as pd
import sympy
import NsriMain_data
mpl.use('TkAgg')

from scipy.special import hyp2f1
from scipy.special import gamma
from scipy.special import spherical_jn
from scipy.special import sph_harm
from scipy.special import poch
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

#distribution function

def DF(q,kappa,epsilon,Jr,J2):
    return (3*gamma(6-q)/(2*np.power(2*np.pi,5/2)))*np.power(-(kappa*Jr+epsilon),7/2-q)*hyp2f1(q/2,-7/2+q,1,-np.power(J2,2)/(2*(kappa*Jr+epsilon)))/(gamma(9/2-q)*gamma(1))

def DF2(q,kappa,epsilon,Jr,J2):
	return (3*gamma(6-q)/(2*np.power(2*np.pi,5/2)))*np.power(-(kappa*Jr+epsilon),7/2-q)*(1/(np.power(-np.power(J2,2)/(2*(kappa*Jr+epsilon)),q/2)))*hyp2f1(q/2,q/2,9/2-q/2,1/(-np.power(J2,2)/(2*(kappa*Jr+epsilon))))/(gamma(1-q/2)*gamma(9/2-q/2))

#main function
def SrsMain():

	#initial condition
	L = 0.027
	G_M=1
	q=float(NsriMain_data.q)


	"""
	#calculation of r_h
	r_h_solve=sympy.Symbol('r_h_solve')
	r_h=sympy.solve(np.power(L,2)/np.power(r_h_solve,3)+G_M*r_h_solve/np.power(1+np.power(r_h_solve,2),3/2))
	print(r_h)
	"""

	r_h=0.2
	#print(np.power(L,2)/np.power(r_h,3)+G_M*r_h/np.power(1+np.power(r_h,2),3/2))

	Npoints=np.arange(1,7,1)
	list_w_J=[]
	list_D_nl_final=[]
	list_U_nl_final=[]

	for n in Npoints:
	
		k=4
		l=2
		m=2
		t=min(l-m,l+n)

		Jpoints=np.arange(0,n+1,1)
		Ipoints=np.arange(0,k+1,1)
		Upoints=np.arange(-80,80,1)
		Rpoints=1/2+(3/4)*Upoints*0.01-(1/4)*np.power(Upoints*0.01,3)	
	
		#potential is Plummer model
		Psi=-G_M/np.sqrt(1+np.power(r_h,2))
		D2r_Psi=G_M/np.power(1+np.power(r_h,2),3/2)-3*G_M*np.power(r_h,2)/np.power(1+np.power(r_h,2),5/2)

		
		DDF=0.0
		DFpoints=[]

		J2=L

		epsilon=0.5*np.power(J2,2)/np.power(r_h,2)-Psi
		kappa=np.sqrt((3*np.power(J2,2)/np.power(r_h,4))-D2r_Psi)
		Psi_r=-G_M/np.sqrt(1+np.power(Rpoints,2))

		omega1=kappa
		omega2=L/np.power(0.5,2)

		Djr_DF=0
		Djr_DFpoints=[]
		Dj2_DF=0
		Dj2_DFpoints=[]
		Dj_DF=0
		Dj_DFpoints=[]

		diff_DF=[]
		diff_Jr=[]

		diff_J2=L

		Djr_H=0
		Djr_Hpoints=[]
		Dj2_H=0
		Dj2_Hpoints=[]
		Dj_H=0
		Dj_Hpoints=[]
		Ppoints_partial_list=[]
		Qpoints_partial_list=[]
		E_List=[]


		eta=0.074
		omega=0.6+eta*1j
		Jrpoints=(0.5*Psi_r-epsilon)/omega1

		for Jr in Jrpoints:	
			if -np.power(J2,2)/(2*(kappa*Jr+epsilon))<1:	
				DDF = DF(q,kappa,epsilon,Jr,J2)
			else:
				DDF = DF2(q,kappa,epsilon,Jr,J2)
		
			DFpoints.append(DDF)
			Djr_H=(4*Jr+2*(J2+np.sqrt(np.power(J2,2)+4)))/(2*np.power(Jr,2)+2*Jr*(J2+np.sqrt(np.power(J2,2)+4))+np.power(J2,2)+J2*np.sqrt(np.power(J2,2)+4)+2)
			Djr_Hpoints.append(Djr_H)
			Dj2_H=(1/2)*(Jr+(J2*Jr)/np.sqrt(np.power(J2,2)+4)+J2+(1/2)*(np.sqrt(np.power(J2,2)+4)+np.power(J2,2)/np.sqrt(np.power(J2,2)+4)))/np.power(np.power(Jr,2)+Jr*(J2+np.sqrt(np.power(J2,2)+4))+(1/2)*(np.power(J2,2)+2+J2*np.sqrt(np.power(J2,2)+4)),2)
			Dj2_Hpoints.append(Dj2_H)
			Ppoints_partial=omega-(Djr_H+Dj2_H)
			Ppoints_partial_list.append(Ppoints_partial)
			Qpoints_partial=DDF/(omega-(Djr_H+Dj2_H))
			Qpoints_partial_list.append(Qpoints_partial)

		diff_DF=np.ediff1d(DFpoints)
		diff_Jr=np.ediff1d(Jrpoints)
		Djr_DF=diff_DF/diff_Jr
		Dj2_DF=0

	

		Ppoints_partial_list=(Djr_DF+Dj2_DF)/Ppoints_partial_list[:-1]
		Ppoints_integrate=integrate.simps(Ppoints_partial_list,Jrpoints[:-1])

		Qpoints_integrate=integrate.simps(Qpoints_partial_list,Jrpoints)
	

		#calculation of U_nl

		P=np.sqrt(((2*k+l+2*n+1/2)*gamma(2*k+l+n+1/2)*gamma(l+n+1/2))/(gamma(2*k+n+1)*np.power(gamma(l+1),2)*gamma(n+1)))

		alpha_U_nl_list=[]
		r_U_nl=np.ones((k+1,n+1))

		for i in Ipoints:
			for j in Jpoints:
				alpha_U_nl = poch(-k,i)*poch(l+1/2,i)*poch(2*k+l+n+1/2,j)*poch(i+l+1/2,j)*poch(-n,j)/(poch(l+1,i)*poch(1,i)*poch(l+i+1,j)*poch(l+1/2,j)*poch(1,j))
				alpha_U_nl_list.append(alpha_U_nl)

		alpha_U_nl_list=np.reshape(alpha_U_nl_list,(k+1,n+1))

		U_nl=np.ones((k+1,n+1))

		U_nl_list=[]
		U_nl_r_list=[]
		U_nl_final=[]

	
		for z in Rpoints:
			for i in Ipoints:
				for j in Jpoints:
					r_U_nl[i][j]=np.power(z,2*i+2*j)
					U_nl[i][j] = alpha_U_nl_list[i][j]*r_U_nl[i][j]
			U_nl_r=np.sum(U_nl)
			U_nl_r_list.append(U_nl_r)


		U_nl_final=(-1)*P*np.multiply(np.power(Rpoints,l),U_nl_r_list)
		list_U_nl_final.append(U_nl_final)

		#calculation of D_nl

		beta_D_nl=np.ones(n+1)
		r_D_nl=np.ones(n+1)
		beta_D_nl_list=[]

		S=(gamma(k+1)/(np.pi*gamma(2*k+1)*gamma(k+1/2)))*np.sqrt(((2*k+l+2*n+1/2)*gamma(2*k+n+1)*gamma(2*k+l+n+1/2))/(gamma(l+n+1/2)*gamma(n+1)))
	
		for z in Rpoints:
			for j in Jpoints:
				r_D_nl[j] = np.power(1-np.power(z,2),j)
				beta_D_nl[j] = poch(2*k+l+n+1/2,j)*poch(k+1,j)*poch(-n,j)/(poch(2*k+1,j)*poch(k+1/2,j)*poch(1,j))
			beta_D_nl_list.append(np.sum(np.multiply(r_D_nl,beta_D_nl)))

	
		D_nl_final=[]
		
		D_nl_final.append(np.power(-1,n)*S*np.power(1-np.power(Rpoints,2),k-1/2)*np.power(Rpoints,l)*beta_D_nl_list)
		#print(D_nl_final)


		list_D_nl_final.append(np.array(D_nl_final))

		RRpoints=Rpoints
		
		u0=0.01
		u1=1.0
	
	
		#angle
		Dr_theta1=0.0
		Dr_theta2=0.0
	
		E_energy=0.5*Psi_r	

		Dr_theta1points=[]
		Dr_theta2points=[]
	
		#potential is plummer model
		Dr_theta1points=omega1/np.sqrt(2*(E_energy-Psi_r)-np.power(J2/Rpoints,2))
		Dr_theta2points=(omega2-J2/np.power(Rpoints,2))/np.sqrt(2*(E_energy-Psi_r)-np.power(J2/Rpoints,2))
		theta1points=Dr_theta1points*(3/4-(3/4)*np.power(Upoints*0.01,2))
		theta2points=Dr_theta2points*(3/4-(3/4)*np.power(Upoints*0.01,2))
	
		#angle coefficients
		n1=1.0
		n2=1.0

		Du_W_J=(1/np.pi)*(3/4-(3/4)*np.power(Upoints*0.01,2))*Dr_theta1points*U_nl_final*np.cos(n1*np.sum(theta1points)+n2*np.sum(theta2points))
		w_J=integrate.simps(Du_W_J,Upoints)
		list_w_J.append(w_J)

	#end n loop

	print(len(Rpoints))


	"""
	#showing D_r
	plt.plot(Rpoints.reshape(160,1),list_D_nl_final[0].flatten(),marker='.',color="blue")
	plt.plot(Rpoints,list_D_nl_final[1].flatten(),marker='.',color="green")
	plt.plot(Rpoints,list_D_nl_final[2].flatten(),marker='.',color="purple")
	plt.plot(Rpoints,list_D_nl_final[3].flatten(),marker='.',color="navy")
	plt.plot(Rpoints,list_D_nl_final[4].flatten(),marker='.',color="orange")
	plt.plot(Rpoints,list_D_nl_final[5].flatten(),marker='.',color="red")

	plt.xlabel("r")
	plt.ylabel("D_nl")

	plt.show()
	"""
	"""
	#showing U_nl
	plt.plot(Rpoints,list_U_nl_final[0],marker='.',color="blue")
	plt.plot(Rpoints,list_U_nl_final[1],marker='.',color="green")
	plt.plot(Rpoints,list_U_nl_final[2],marker='.',color="purple")
	plt.plot(Rpoints,list_U_nl_final[3],marker='.',color="navy")
	plt.plot(Rpoints,list_U_nl_final[4],marker='.',color="orange")
	plt.plot(Rpoints,list_U_nl_final[5],marker='.',color="red")
	
	plt.xlabel("r")
	plt.ylabel("U_nl")

	plt.show()
	"""

	#print(list_w_J)


	#rotaton matrix

	n2_rv=1

	Rotpoints=[]
	list_Rotpoints=[]
	Ipoints=np.arange(0,314,1)
	Tpoints=np.arange(0,t+1,1)
	for t in Tpoints:
		for i in Ipoints:
			Rotpoints.append(np.power(-1,t)*np.sqrt(math.factorial(l+n2_rv)*math.factorial(l-n2_rv)*math.factorial(l+m)*math.factorial(l-m))*np.power(np.cos(i*0.01/2),2*l+n2_rv-m-2*t)*np.power(np.sin(i*0.01/2),2*t+m-n2_rv)/(math.factorial(l-m-t)*math.factorial(l+n2_rv-t)*math.factorial(t)*math.factorial(t+m-n2_rv)))

	Rotpoints=np.reshape(Rotpoints,(t+1,314))
	Rotpoints=np.sum(Rotpoints, axis=0)


	#P_omega and Q_omega

	omega_points=np.arange(0,1,0.01)


	I_w=0
	J_w=0
	
	Ppoints=np.ones((6,6),dtype=np.complex128)
	Qpoints=np.ones((6,6),dtype=np.complex128)
	I_wpoints=[]
	J_wpoints=[]



	for i in range(len(list_w_J)):
		for j in range(len(list_w_J)):
			Ppoints[i][j]=L*list_w_J[i]*list_w_J[j]*Ppoints_integrate
			Qpoints[i][j]=list_w_J[i]*list_w_J[j]*Qpoints_integrate

	I_wpoints_partial=0
	I_wpoints_partial_points=[]

	I2points=np.arange(0,314,1)

	for i2 in I2points:
		I_wpoints_partial=np.sin(i2*0.01)*np.sign(np.cos(i2*0.01))*Rotpoints[i2]*Rotpoints[i2]
		I_wpoints_partial_points.append(I_wpoints_partial)

	I_wpoints=np.power(2*np.pi,3)*sph_harm(2,l,0,np.pi/2)*sph_harm(2,l,0,np.pi/2)*sum(I_wpoints_partial_points)
	J_wpoints=2*np.power(2*np.pi,3)*sph_harm(2,l,0,np.pi/2)*sph_harm(2,l,0,np.pi/2)*Rotpoints[157]*Rotpoints[157]


	#M_omega

	C=2*np.power(2*np.pi,3)*np.power(abs(sph_harm(2,l,0,np.pi/2)),2)/(2*l+1)

	M0_omega=C*np.array(Ppoints)
	M1_omega=np.array(Ppoints)*I_wpoints.real+np.array(Qpoints)*J_wpoints.real

	M0_omega_norm=np.abs(M0_omega)
	M1_omega_norm=np.abs(M1_omega)

	M_omega=M0_omega+float(NsriMain_data.alpha)*M1_omega


	#M_omegaの描写

	fig, axes=plt.subplots(1,2)
	fig.subplots_adjust(wspace=0.4)

	sns.heatmap(M0_omega_norm,cmap='Blues',square=True,ax=axes[0])
	sns.heatmap(M1_omega_norm,cmap='Blues',square=True,ax=axes[1])

	plt.show()


	#eigenvalue,eigenvector of M_omega

	eigen_value,eigen_vector=LA.eig(M_omega)


	#calculation of density perturbation

	NNpoints=np.arange(0,6,1)
	Density_PP=[]
	list_Density_P=[]
	Density_P=[]

	for i in NNpoints:
		Density_PP=eigen_vector[i][2]*list_D_nl_final[i]
		list_Density_P.append(Density_PP)

	Density_P=np.sum(list_Density_P,axis=0)


	#z=0におけるx,y平面上の密度振動成分

	def orbital(m,l,r,p,t):
		d=np.real(sph_harm(m,l,p,t)*Density_P)
		#d=np.real(sph_harm(m,l,p,t))
		x=r*np.sin(t)*np.cos(p)
		y=r*np.sin(t)*np.sin(p)
		return x,y,d


	#FigureとAxes
	cm=plt.cm.get_cmap('PRGn')
	fig=plt.figure(figsize=(8,6))
	ax=fig.add_subplot(111)
	ax.set_aspect('equal')

	ax.set_xlabel("x",size=16)
	ax.set_ylabel("y",size=16)



	#phiの格子点データ
	p=np.linspace(0,2*np.pi,160)
	r=Rpoints

	t=np.pi*(1/2)

	#格子点の作成
	rr,pp=np.meshgrid(r,p)

	# l=2,m=2軌道,m,l,pp,t
	x,y,d=orbital(2,2,rr,pp,t)

	#軌道を描画
	mappable=ax.scatter(x,y,c=d,cmap=cm)
	fig.colorbar(mappable,ax=ax)
	plt.show()



	#y=0におけるx,z平面上の密度振動成分


	def orbital(m,l,r,p,t):
		d=np.real(sph_harm(m,l,p,t)*Density_P)
		x=r*np.sin(t)*np.cos(p)
		z=r*np.cos(t)
		return x,z,d


	#FigureとAxes
	cm=plt.cm.get_cmap('PRGn')
	fig=plt.figure(figsize=(8,6))
	ax=fig.add_subplot(111)
	ax.set_aspect('equal')


	ax.set_xlabel("x",size=16)


	#phiの格子点データ
	p=0
	t=np.linspace(0,2*np.pi,160)
	r=Rpoints


	#格子点の作成
	rr,tt=np.meshgrid(r,t)

	# l=2,m=2軌道,m,l,pp,t
	x,z,d=orbital(2,2,rr,p,tt)

	#軌道を描画
	mappable=ax.scatter(x,z,c=d,cmap=cm)
	fig.colorbar(mappable,ax=ax)
	plt.show()

	"""
	#3次元の密度振動成分

	def orbital(m,l,r,p,t):
		d=np.real(sph_harm(m,l,p,t)*Density_P)
		#d=np.real(sph_harm(m,l,p,t))
		x=r*np.sin(t)*np.cos(p)
		y=r*np.sin(t)*np.sin(p)
		z=r*np.cos(t)
		return x,y,z,d


	#FigureとAxes
	cm=plt.cm.get_cmap('PRGn')
	fig=plt.figure(figsize=(8,6))
	ax=fig.add_subplot(111,projection='3d')
	ax.set_box_aspect((1,1,1))


	ax.set_xlabel("x",size=16)
	ax.set_ylabel("y",size=16)
	ax.set_zlabel("z",size=16)

	#アングルを設定
	ax.view_init(45,45)

	#theta,phiの格子点データ
	r=Rpoints
	t=np.linspace(0,np.pi,160)
	p=np.linspace(0,2*np.pi,160)


	#格子点の作成
	tt,pp,rr=np.meshgrid(t,p,r)

	# l=2,m=2軌道 ,m,l,pp,tt
	x,y,z,d=orbital(2,2,rr,pp,tt)

	#軌道を描画
	mappable=ax.scatter(x,y,z,c=d,cmap=cm,s=0.01)
	fig.colorbar(mappable,ax=ax)

	plt.show()
	"""






