import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import cmath
import seaborn as sns
import pandas as pd
import sympy
import os
import sys
import srs
import Nbpkivy

f=open(os.devnull, 'w')
sys.stderr = f
sys.stdout = f
sys.stdin = f


import requests
import webbrowser
import NsriMain_data

from scipy.special import hyp2f1
from scipy.special import gamma
from scipy.special import spherical_jn
from scipy.special import sph_harm
from scipy.special import poch
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from scipy.stats import multivariate_normal

from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.floatlayout import FloatLayout
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.screenmanager import ScreenManager,Screen
from kivy.uix.boxlayout import BoxLayout

from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')
Config.set('graphics', 'width', '640')
Config.set('graphics', 'height', '480')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty
from kivy.graphics import Ellipse,Color
from kivy.uix.button import Button
import kivy


class Display(BoxLayout):
	pass

"""
class CircularButton2(Button):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.background_color=[0,0,1,1]
		self.background_normal=''
	def draw(self,*args):
		self.canvas.clear()
		with self.canvas:
			Color(0, 1, 0, 1) 
			d = min(self.size)
			Ellipse(pos=self.pos, size=(d,d))
"""

class Button2(Button):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.background_color=[1,1,1,1]
		self.background_normal=''

class Button3(Button):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.background_color=[0.25,0.41,0.88,1]
		self.background_normal=''

class Button4(Button):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.background_color=[0.39,0.58,0.93,1]
		self.background_normal=''

class SsWidget(Screen):
	# ボタンをクリック時
	def on_command1(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/8figure.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)		
	def on_command2(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/globularcluster.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)
	def on_command3(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/ComplexPlane.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command4(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/binarystar.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command5(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/blackhole.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command6(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/gravitational-wave.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)
	def on_command7(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/transfer-angularmomentum.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)		
	def on_command8(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/SRS-Plummer.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)
	def on_command9(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/VanDelPol-equation.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command10(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/edu20170722.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command11(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/benardconvection.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command12(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/randomwalk.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)
	def on_command13(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/langevin.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)		
	def on_command14(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/Duffing-equation.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)
	def on_command15(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/hr.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command16(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/friedmann.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command17(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/Maxplank.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command18(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/coulomb.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)
	def on_command19(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/cyclotron.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)		
	def on_command20(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/aurora.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)
	def on_command21(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/NGC104-velocity-space.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command22(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/Jeans-equation.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command23(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/Hall-MHD.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
	def on_command24(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/lindblad-resonance.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)
	def on_command25(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/Polarization.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)		
	def on_command26(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/Complex-velocity-potential.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)
	def on_command27(self, **kwargs):
		url="https://natural-science-research-institute.org/menu/edu/Linearization-equation.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)	
		
	
	

class NewsWidget(Screen):
	# ボタンをクリック時
	def on_command7(self, **kwargs):
		url="https://natural-science-research-institute.org/news.html"
		browser=webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
		browser.open(url,new=1,autoraise=True)


class SrsWidget(Screen):
	# プロパティの追加
	text = StringProperty()
	
	# ボタンをクリック時
	def on_command(self, **kwargs):
		if float(self.ids.alpha.text) <= 1.0 and float(self.ids.alpha.text) >= 0 and float(self.ids.q.text) >= -16 and float(self.ids.q.text) <= 2.0:
			self.text = "[color=ffffff]OK.[/color]" 
			NsriMain_data.alpha=self.ids.alpha.text
			NsriMain_data.q=self.ids.q.text
			srs.SrsMain()
		else:
			self.text='[color=b0e0e6]Invallid value.Please input value again.[/color]'

class NbpWidget(Screen):
	# プロパティの追加
	text = StringProperty()
	# ボタンをクリック時
	def on_command(self, **kwargs):
		if abs(float(self.ids.text1.text)) <= 2.0 and abs(float(self.ids.text2.text)) <= 2.0 and abs(float(self.ids.text3.text)) <= 2.0 and abs(float(self.ids.text4.text)) <= 2.0 and abs(float(self.ids.text5.text)) <= 2.0 and abs(float(self.ids.text6.text)) <= 2.0 and abs(float(self.ids.text7.text)) <= 2.0 and abs(float(self.ids.text8.text)) <= 2.0 and abs(float(self.ids.text9.text)) <= 2.0 and abs(float(self.ids.text10.text)) <= 2.0 and abs(float(self.ids.text11.text)) <= 2.0 and abs(float(self.ids.text12.text)) <= 2.0:
			self.text="[color=ffffff]OK.[/color]"
			NsriMain_data.X10 = self.ids.text1.text
			NsriMain_data.Y10 = self.ids.text2.text
			NsriMain_data.X20 = self.ids.text3.text
			NsriMain_data.Y20 = self.ids.text4.text
			NsriMain_data.X30 = self.ids.text5.text
			NsriMain_data.Y30 = self.ids.text6.text
			NsriMain_data.X1d0 = self.ids.text7.text
			NsriMain_data.Y1d0 = self.ids.text8.text
			NsriMain_data.X2d0 = self.ids.text9.text
			NsriMain_data.Y2d0 = self.ids.text10.text
			NsriMain_data.X3d0 = self.ids.text11.text
			NsriMain_data.Y3d0 = self.ids.text12.text
			Nbpkivy.NbpMain()
		else:
			self.text='[color=b0e0e6]Invallid value.Please input value again.[/color]'

		

class MyNsriMain2App(App):
	def __init__(self, **kwargs):
		super(MyNsriMain2App, self).__init__(**kwargs)
		# ウィンドウのタイトル名
		self.title = 'Yamaguchi\'s natural science research institute'

	def build(self):
		return Display()

if __name__ == '__main__':
	MyNsriMain2App().run()







