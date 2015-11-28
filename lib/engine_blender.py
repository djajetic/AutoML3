# Linear enseble of best models
#Damir Jajetic, 2015
from sklearn.externals import joblib
from sklearn import linear_model, naive_bayes, neighbors, cross_validation, feature_selection
from sklearn import metrics, ensemble, decomposition, preprocessing, svm, manifold, mixture, neural_network
from sklearn import cross_decomposition, naive_bayes, neighbors, kernel_approximation, random_projection, isotonic
import libscores
import multiprocessing
import time
import shutil
import os
import numpy as np
import data_io
import psutil
import data_converter
import copy
from sklearn.utils import shuffle
from operator import itemgetter
from sklearn.pipeline import Pipeline
from scipy import stats
import itertools


class no_transform:
	def fit_transform(self, preds):
		return preds
			
def blend2(x1,x2,y, metric, x1valid, x2valid, x1test, x2test):
	#mm = preprocessing.MinMaxScaler()
	mm = no_transform()
	mbest_score = 0
	for w1 in np.arange(0, 1.1, 0.1):
		w2 = 1- w1
		x = mm.fit_transform(x1)*w1  +  mm.fit_transform(x2)*w2
		x = np.clip(x, 0,1)
		exec('score = libscores.'+ metric  + '(y, x)')
		if score > mbest_score:
			mbest_score = score
			mbest_w1 = w1
			mbest_x  = x
	mbest_w2 = 1- mbest_w1
	xvalid = mm.fit_transform(x1valid) * mbest_w1 +  mm.fit_transform(x2valid)* mbest_w2
	xtest =  mm.fit_transform(x1test) * mbest_w1 +  mm.fit_transform(x2test) * mbest_w2

	return mbest_score, xvalid, xtest
	
def blend3(x1,x2, x3, y, metric, x1valid, x2valid, x3valid, x1test, x2test, x3test):
	#mm = preprocessing.MinMaxScaler()
	mm = no_transform()
	mbest_score = 0
	for w1 in np.arange(0.2, 0.5, 0.1):
		for w2 in np.arange(0.1, 0.5, 0.1):
			w3 = 1- w1 - w2

			if w3 > 0:
				x = mm.fit_transform(x1)*w1  +  mm.fit_transform(x2)*w2 +  mm.fit_transform(x3)*w3
				x = np.clip(x, 0,1)
				exec('score = libscores.'+ metric  + '(y, x)')
				if score > mbest_score:
					mbest_score = score
					mbest_w1 = w1
					mbest_w2 = w2
	
	mbest_w3 = 1- mbest_w1- mbest_w2
	xvalid = mm.fit_transform(x1valid) * mbest_w1 +  mm.fit_transform(x2valid)* mbest_w2 +  mm.fit_transform(x3valid)* mbest_w3
	xtest =  mm.fit_transform(x1test) * mbest_w1 +  mm.fit_transform(x2test) * mbest_w2 +  mm.fit_transform(x3test) * mbest_w3

	return mbest_score, xvalid, xtest

	#in original version there was also unused "blend4" function, for declutttering this was removed, but still exists on codalab public submissions
	
def blender (sd, srd, Nworkers, stop_writing, output_dir, basename, Lstart, Ltime_budget, Lfold):
	try:
		split = int(len(sd.LD.data['Y_train'])*0.5)
		cycle = 1 #cycle 0 is all zeros
		best_score = 0
		atbest = 0
		
		while(1):
			try:
				time.sleep(0.5)
				# limit to 100 predictions
				if cycle >  (time.time() - Lstart)/Ltime_budget * 100:
					time.sleep(1)
					continue
				
				temp_workers_data = []
				workers_data = []
				for wr_no in range(Nworkers):
					exec("wr_data =  sd.worker"+str(wr_no))
					if wr_data['done'] > 0:
						temp_workers_data.append(wr_data)
				wgroups = [i['blend_group'] for i in temp_workers_data]
				for group in np.unique(wgroups):
					twdata = [i for i in temp_workers_data if i['blend_group'] == group]
					twdata = sorted(twdata, key=itemgetter('score'), reverse=True)
					
					workers_data.append(twdata[0])
					try:
						workers_data.append(twdata[1])
					except:
						pass
					print group, len(twdata), len(workers_data)
					
				
				# this is patch for codalab VM
				workers_data_raw = []
				raw0_data =  srd.raw_model
				if raw0_data['done'] ==1:
					workers_data_raw.append(raw0_data)
					
				raw1_data =  srd.raw_model1
				if raw1_data['done'] ==1:
					workers_data_raw.append(raw1_data)
					
				raw2_data =  srd.raw_model2
				if raw2_data['done'] ==1:
					workers_data_raw.append(raw2_data)
					
				raw3_data =  srd.raw_model3
				if raw3_data['done'] ==1:
					workers_data_raw.append(raw3_data)
				
				raw4_data =  srd.raw_model4
				if raw4_data['done'] ==1:
					workers_data_raw.append(raw4_data)

				
				if len(workers_data_raw) > 0:
					workers_data_raw = sorted(workers_data_raw, key=itemgetter('score'), reverse=True)
					workers_data.append(workers_data_raw[0])
					try:
						workers_data.append(workers_data_raw[1])
					except:
						pass
					try:
						workers_data.append(workers_data_raw[2])
					except:
						pass
				
				workers_data = sorted(workers_data, key=itemgetter('score'), reverse=True)
				
				if len(workers_data) > 0:
					worker0 = workers_data[0]
					preds_valid = worker0['preds_valid'] 
					preds_test = worker0['preds_test'] 
					
					y = sd.yt_raw[split:]
					if Lfold  > 1:
						y = sd.yt_raw
									
					x = worker0['preds_2fld']
					
					exec('s0 = libscores.'+ sd.LD.info['metric']  + '(y, x)')
					best_score = s0
					
					
					#short run can't wait for blend (usable only for AutoML 1)
					try:
						if s0 > atbest and cycle < 2:
							atbest = best_score * 0.9 #not reilable score
							if sd.LD.info['target_num']  == 1:
								preds_valid = preds_valid[:,1]
								preds_test = preds_test[:,1]
									
							preds_valid = np.clip(preds_valid,0,1)
							preds_test = np.clip(preds_test,0,1)
							filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
							data_io.write(os.path.join(output_dir,filename_valid), preds_valid)
							filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
							data_io.write(os.path.join(output_dir,filename_test), preds_test)
						
							cycle += 1
					except:
						pass
						

					if Lfold < 4:
						Lsample = 4
					else:
						Lsample = 6
					xa = 0
					Lssample = Lsample - 1
					
					for iter_worker in itertools.combinations(workers_data[:Lsample], 2):
						xa = xa+1
						worker0 = iter_worker[0]
						worker1 = iter_worker[1]
						s01, validt, testt = blend2(worker0['preds_2fld'],worker1['preds_2fld'],y, sd.LD.info['metric'] , 
										    worker0['preds_valid'], worker1['preds_valid'], 
										    worker0['preds_test'], worker1['preds_test'])
					
						if s01 > best_score:
							best_score = s01
							preds_valid = validt
							preds_test = testt
						
					xa = 0
					
					for iter_worker in itertools.combinations(workers_data[:Lssample], 3):
						xa = xa+1
						worker0 = iter_worker[0]
						worker1 = iter_worker[1]
						worker2 = iter_worker[2]
						s012, validt, testt = blend3(worker0['preds_2fld'],worker1['preds_2fld'],worker2['preds_2fld'],y, sd.LD.info['metric'] , 
										    worker0['preds_valid'], worker1['preds_valid'], worker2['preds_valid'], 
										    worker0['preds_test'], worker1['preds_test'], worker2['preds_test'])
						if s012 > best_score:
							best_score = s012
							preds_valid = validt
							preds_test = testt
						

					if stop_writing.is_set() == False: #until last 10 seconds (event signal)
						if best_score > atbest:
							atbest = best_score
							print "naj =", workers_data[0]['score'] , best_score, atbest
							
							if  sd.LD.info['target_num']  == 1:
								preds_valid = preds_valid[:,1]
								preds_test = preds_test[:,1]
								
							preds_valid = np.clip(preds_valid,0,1)
							preds_test = np.clip(preds_test,0,1)
							filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
							data_io.write(os.path.join(output_dir,filename_valid), preds_valid)
							filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
							data_io.write(os.path.join(output_dir,filename_test), preds_test)
						
							cycle += 1
					else:
						print 'stop writing is set'
					
			except Exception as e:
				print 'exception in blender process' + '     ' +  str(e)
				# in case of any problem, let's try again
	except Exception as e:
				print 'exception in blender main process' + '     ' +  str(e)
		
