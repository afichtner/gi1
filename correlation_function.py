# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import green as g
import source as s
import parameters
import get_propagation_corrector as gpc

def correlation_function(rec0=0,rec1=1,effective=0,plot=0):

	"""
	cct, t, ccf, f = correlation_function(rec0=0,rec1=1,effective=0,plot=0)

	Compute time- and frequency-domain correlation functions. 


	INPUT:
	------
	rec0, rec1:		indeces of the receivers used in the correlation. 
	plot:			When plot=1, the source distribution, and the time- and frequency domain correlation functions are plotted.
	effective:		When effective==1, effective correlations are computed using the propagation correctors stored in OUTPUT/correctors.
					The source power-spectral density is then interpreted as the effective one.

	OUTPUT:
	-------
	cct, t:		Time-domain correlation function and time axis [N^2 s / m^4],[s].
	ccf, f:		Frequency-domain correlation function and frequency axis [N^2 s^2 / m^4],[1/s].

	Last updated: 27 May 2016.
	"""

	#==============================================================================
	#- Initialisation.
	#==============================================================================

	p=parameters.Parameters()

	x_line=np.arange(p.xmin,p.xmax,p.dx)
	y_line=np.arange(p.ymin,p.ymax,p.dy)

	x,y=np.meshgrid(x_line,y_line)

	f=np.arange(p.fmin-p.fwidth,p.fmax+p.fwidth,p.df)
	omega=2.0*np.pi*f

	t=np.arange(p.tmin,p.tmax,p.dt)

	#- Frequency- and space distribution of the source. ---------------------------

	S,indices=s.space_distribution(plot)
	instrument,natural=s.frequency_distribution(f)
	filt=natural*instrument*instrument

	#- Read propagation corrector if needed. --------------------------------------

	if (effective==1):

		gf=gpc.get_propagation_corrector(rec0,rec1,plot=0)

	else:

		gf=np.ones(len(f),dtype=complex)

	#==============================================================================
	#- Compute inter-station correlation function.
	#==============================================================================

	cct=np.zeros(np.shape(t),dtype=float)
	ccf=np.zeros(np.shape(f),dtype=complex)

	for idf in range(len(omega)):

		P=g.conjG1_times_G2(p.x[rec0],p.y[rec0],p.x[rec1],p.y[rec1],x,y,omega[idf],p.dx,p.dy,p.rho,p.v,p.Q)
		ccf[idf]=gf[idf]*np.conj(np.sum(P*S))

		cct=cct+np.real(filt[idf]*ccf[idf]*np.exp(1j*omega[idf]*t))

	cct=cct*p.dx*p.dy*p.df

	#==============================================================================
	#- Plot result.
	#==============================================================================

	if (plot==1):

		#- Frequency domain.
		plt.semilogy(f,np.abs(ccf),'k')
		plt.semilogy(f,np.real(ccf),'b')
		plt.title('frequency-domain correlation function (black=abs, blue=real)')
		plt.xlabel('frequency [Hz]')
		plt.ylabel('correlation [N^2 s^2/m^4]')
		plt.show()

		#- Time domain.

		tt=np.sqrt((p.x[rec0]-p.x[rec1])**2+(p.y[rec0]-p.y[rec1])**2)/p.v
		cct_max=np.max(np.abs(cct))

		plt.plot(t,cct,'k',linewidth=2.0)
		plt.plot([tt,tt],[-1.1*cct_max,1.1*cct_max],'--',color=(0.5,0.5,0.5),linewidth=1.5)
		plt.plot([-tt,-tt],[-1.1*cct_max,1.1*cct_max],'--',color=(0.5,0.5,0.5),linewidth=1.5)

		plt.ylim((-1.1*cct_max,1.1*cct_max))
		plt.title('correlation function')
		plt.xlabel('time [s]')
		plt.ylabel('correlation [N^2 s/m^4]')
		plt.show()

	#==============================================================================
	#- Return.
	#==============================================================================

	return cct, t, ccf, f