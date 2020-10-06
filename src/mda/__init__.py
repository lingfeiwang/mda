#!/usr/bin/python3

import numpy as np

def mda(data,group,datatype='raw',**ka):
	"""Computes the Maximal Discriminating Axes (MDA) between cell groups in scRNA-seq.
	
	Parameters
	------------
	data:		numpy.ndarray(shape=(n_gene,n_cell))
		Gene expression matrix. Can be raw read counts or log(CPM+1).
	group:		numpy.ndarray(shape=(n_cell,))
		Group ID of each cell. Each group must have at least 2 cells.
		Values must cover 0 to n_group-1.
	datatype:	str
		Type of data.
		
		* raw:	Raw read counts
		
		* lcpm:	Log(CPM+1). Natural log.
	ka:			dict
		Keyword arguments passed to sklearn.discriminant_analysis.LinearDiscriminantAnalysis
		
	Returns
	--------
	loc:	numpy.ndarray(shape=(n_cell,n_group-1))
		Coordinates of each cell on the (n_group-1) dimensional MDA.
	prob:	numpy.ndarray(shape=(n_cell,n_group))
		Probability of each cell being assigned to each group using coordinates on MDA.

	"""
	import numpy as np
	from collections import Counter
	from scipy.linalg import orth
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda0
	#Initial QC
	if data.ndim!=2:
		raise TypeError('data must have 2 dimensions.')
	ngene,ncell=data.shape
	if ngene==0 or ncell==0:
		raise ValueError('Empty data matrix.')
	if not np.isfinite(data).all():
		raise ValueError('data matrix must be finite.')
	if data.min()<0:
		raise ValueError('data matrix must be non-negative.')
	if group.shape!=(ncell,):
		raise ValueError('Unmatched cell count in parameters data and group.')
	t1=np.sort(np.unique(group))
	ngroup=len(t1)
	if ngroup<2:
		raise ValueError('Must have at least two groups.')
	if ngene<ngroup:
		raise ValueError('Gene count must be no fewer than group count.')
	if not (t1==np.arange(ngroup)).all():
		raise ValueError('Group must be 0 to n_group-1.')
	t1=np.min(list(Counter(group).values()))
	if t1<2:
		raise ValueError('Each group must have at least two cells.')
	#Conversion to log(CPM+1), i.e. lcpm
	if datatype=='raw':
		data=np.log(1+1E6*data/data.sum(axis=0))
	elif datatype=='lcpm':
		pass
	else:
		raise ValueError('Unknown parameter datatype {}. Must be one of "raw", "lcpm".'.format(datatype))
	#Compute unit vectors for MDA
	vecs=np.array([data[:,group==x].mean(axis=1) for x in range(ngroup)])
	vecs=vecs[1:]-vecs[0]
	vecs=orth(vecs.T).T
	#Compute coordinates for each cell
	loc=data.T@vecs.T
	#Compute probabilities for each cell with LDA
	lda=lda0(**ka)
	lda.fit(loc,group)
	prob=lda.predict_proba(loc)
	return (loc,prob)

def sphere2tri1(d):
	"""Converts 3-dimensional sqrt probabilities to two-dimensional coordinates within unit triangle.
	
	Uses stereographic projection from (0,0,-1).
	
	Parameters
	------------
	d:	numpy.ndarray(shape=(3,n_cell))
		Input 3-dimensional probabilities
		
	Returns
	--------
	numpy.ndarray(shape=(2,n_cell)):
		Two-dimensional coordinates within unit triangle at [0,0],[1,0],[0.5,sqrt(3)/2]
		
	"""
	import numpy as np
	if d.ndim!=2 or d.shape[0]!=3:
		raise ValueError('Incorrect shape for d.')
	dsx=d[0]/(1+d[2])
	dsy=d[1]/(1+d[2])
	dsr=np.sqrt(dsx**2+dsy**2)
	dst=np.arctan(dsy/dsx)
	dst*=2./3
	dsr*=np.sin(np.pi*2/3)/np.sin(np.pi*2/3-dst)
	dsx=dsr*np.cos(dst)
	dsy=dsr*np.sin(dst)
	return np.array([dsx,dsy])

def sphere2tri(d):
	"""Converts 3-dimensional probabilities to two-dimensional coordinates within unit triangle.
	
	Uses stereographic projection from (0,0,-1), (0,-1,0), (-1,0,0), and then average.
	
	Parameters
	------------
	d:	numpy.ndarray(shape=(3,n_cell))
		Input 3-dimensional probabilities
		
	Returns
	--------
	numpy.ndarray(shape=(2,n_cell)):
		Two-dimensional coordinates within unit triangle at [0,0],[1,0],[0.5,sqrt(3)/2]
		
	"""
	import numpy as np
	if d.ndim!=2 or d.shape[0]!=3:
		raise ValueError('Incorrect shape for d.')
	d=np.sqrt(d)
	ds=np.array([0.5,0.5/np.tan(np.pi/3)])
	dr=np.pi*2/3
	dr=np.array([[np.cos(dr),-np.sin(dr)],[np.sin(dr),np.cos(dr)]])
	d1=sphere2tri1(d)
	d2=np.array([d[1],d[2],d[0]])
	d2=sphere2tri1(d2)
	d2=(np.matmul(dr,(d2.T-ds).T).T+ds).T
	d3=np.array([d[2],d[0],d[1]])
	d3=sphere2tri1(d3)
	d3=(np.matmul(dr.T,(d3.T-ds).T).T+ds).T
	ans=np.array([d1,d2,d3]).mean(axis=0)
	#In top, left, right order
	ans=(np.matmul(dr,(ans.T-ds).T).T+ds).T
	return ans

def draw_triangle(prob,group,group_names,colors=np.array([[1,0,0],[0,1,0],[0,0,1]]),figsize=2,fs=12,**ka):
	"""Draws triangular plot from LDA probabilities between 3 groups.
	
	Parameters
	------------
	prob:			numpy.ndarray(shape=(n_cell,3))
		Probability of each cell being assigned to each group using coordinates on MDA.
	group:			numpy.ndarray(shape=(n_cell,))
		Group ID of each cell. Values must be 0,1,2, matching prob.shape[1].
	group_names:	List of str
		Names of groups.
	colors: 		numpy.ndarray(shape=(3,3))
		Colors in [r,g,b] format for each group (as rows). 0<=r,g,b<=1
	figsize:		float
		Figure size (each dimension)
	fs: 			float
		Font size
	ka:				dict
		Keyword arguments passed to seaborn.kdeplot
		
	Returns
	--------
	matplotlib.pyplot.figure
		Figure drawn
		
	"""
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn
	from matplotlib.colors import LogNorm,to_rgb
	from matplotlib.patches import Polygon
	from matplotlib.colors import LinearSegmentedColormap
	#Plotting parameters
	ka1={'gridsize':50,'shade':True,'shade_lowest':False,'bw':0.05,
		'norm':LogNorm(),'zorder':0}
	ka1.update(ka)
	figsize=(figsize,figsize)
	#Transparency
	alpha0=0.03
	alphabound=[0,0.7]
	
	#Color gradients
	t1=colors
	cs2=[LinearSegmentedColormap('cc'+str(x+1),
		{'red':[[0,t1[x][0],t1[x][0]],[1,t1[x][0],t1[x][0]]],
		'green':[[0,t1[x][1],t1[x][1]],[1,t1[x][1],t1[x][1]]],
		'blue':[[0,t1[x][2],t1[x][2]],[1,t1[x][2],t1[x][2]]],
		'alpha':[[0,alphabound[0],alphabound[0]],[1,alphabound[1],alphabound[1]]]
		}) for x in range(len(colors))]
	
	#Compute coordinates
	d=(sphere2tri(prob.T).T-np.array([0.5,np.sqrt(3)/2])).T
	assert len(d.shape)==2 and d.shape[0]==2
	#Compute levels
	levels1=np.exp(np.linspace(-1,4,40))
	f=plt.figure(figsize=figsize)
	ax=f.add_subplot(111,aspect=1)
	ps=[]
	for xi in range(3):
		#Draw each cell group
		dx,dy=d[:,group==xi]
		seaborn.kdeplot(dx,dy,levels=levels1,cmap=cs2[xi],ax=ax,**ka1)
		ps.append([dx,dy,[colors[xi]]*len(dx)])
	#Frame
	ps=[np.concatenate(x) for x in zip(*ps)]
	t1=np.arange(len(ps[0]))
	np.random.shuffle(t1)
	ax.plot([0,-0.5,0.5,0],[0,-np.sqrt(3)/2,-np.sqrt(3)/2,0],'k-',alpha=0.2,lw=1)
	#Remove plot outside Frame
	ax.add_artist(Polygon([[0,0],[1,-np.sqrt(3)],[5,-5],[5,5],[-1,np.sqrt(3)],[0,0]],facecolor='w',zorder=1))
	ax.add_artist(Polygon([[0,0],[-1,-np.sqrt(3)],[-5,-5],[-5,5],[1,np.sqrt(3)],[0,0]],facecolor='w',zorder=1))
	ax.add_artist(Polygon([[-5,-np.sqrt(3)/2],[5,-np.sqrt(3)/2],[5,-5],[-5,-5],[-5,-np.sqrt(3)/2]],facecolor='w',zorder=1))
	#Texts
	ax.text(0,0.1,group_names[0],ha='center',va='center',fontsize=fs,color=colors[0])
	ax.text(-0.5-0.05,(-1-0.15)*np.sqrt(3)/2,group_names[1],ha='center',va='center',fontsize=fs,color=colors[1])
	ax.text(0.5+0.05,(-1-0.15)*np.sqrt(3)/2,group_names[2],ha='center',va='center',fontsize=fs,color=colors[2])
	ax.axis('off')
	return f












assert __name__ != "__main__"
