# Main function for black box learning
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
import sklearn
from scipy import stats
from scipy import sparse
import engine_worker
import engine_models
import engine_blender
import engine_prep_models
import engine_serial
import engine_preprocess

import sklearn
if sklearn.__version__ >= 0.16:
	from sklearn import ensemble as f16
else:
	try:
		import forest16 as f16 #this is backport of sklearn tree 0.16 version  for 0.15 sklearn created by aad_freiburg
	except:
		from sklearn import ensemble as f16



def write_all_zeros(output_dir, basename, valid_num, test_num, target_num):
	#if something break, to have 0 prediction
	
	preds_valid = np.zeros([valid_num , target_num])
	preds_test = np.zeros([test_num , target_num])
	
	cycle = 0
	filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_valid), preds_valid)
	filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_test), preds_test)



def predict(LD, Loutput_dir, Lstart, Ltime_budget, Lbasename, running_on_codalab):
	try:	
		
		# This code is tuned to be run od Codalab VM and for time and hardware constrained 
		# competition purposes. It could be used for general purpose brute force black box learning,
		# but in that case is should be cleaned of optimizations for HW. 
		
		
		'''
		  If code is run on cadalab, time is more important then proper cross validation
		  in that case first half of (shuffled) test dataset is used for CV
		  when run is not time constrained, then n-fold  could be more approriate.
		  Here is additional 2 fold CV presented, but not used in AutoML2.
		  As in original code CV was not implemented in right way, 4 fold CV is deleted, 
		  for readability of code. But original implementation could be found on Codalab
		  '''
		   
		Lfold = 1
		if running_on_codalab == False:
			Lfold = 2
			print "not running on codalab"
		else:
			print "running on codalab"
		
		
		#for not time constrained run, time budget could be changed
		Ltime_budget *= Lfold

		split = int(len(LD.data['Y_train'])/2) 
		
		#just baseline predictions
		write_all_zeros(Loutput_dir, Lbasename, LD.info['valid_num'], LD.info['test_num'], LD.info['target_num'])
		
		# if dataset is large, and memory and time is constrained (with no swap file), it is better to run on partition of data.
		# Note that this one is risky in this implementation, because it is not stratified. (We can miss some labels, and don't write predictions in right format)
		
		
		try:
			if LD.info['is_sparse'] == 1:
				for acnt in range(5):
					if LD.data['X_train'].data.nbytes > 4*10**9:
						LD.data['X_train']  = LD.data['X_train'] [:len(LD.data['Y_train'])/2]
						LD.data['Y_train']  = LD.data['Y_train'] [:len(LD.data['Y_train'])/2]
			else:
				for acnt in range(5):
					if LD.data['X_train'] .nbytes > 4*10**9:
						LD.data['X_train']  = LD.data['X_train'] [:len(LD.data['Y_train'])/2]
						LD.data['Y_train']  = LD.data['Y_train'] [:len(LD.data['Y_train'])/2]
		except:
			pass
		
		
		try:
			sss = cross_validation.StratifiedShuffleSplit(LD.data['Y_train'], 2, test_size=0.5, random_state=1)
			for train_index, test_index in sss:

				x=LD.data['X_train']
				y=LD.data['Y_train']
				
				X_train, X_test = x[train_index], x[test_index]
				y_train, y_test = y[train_index], y[test_index]
				break
				 
			if LD.info['is_sparse'] == 1:
				LD.data['X_train'] = sparse.vstack([X_train, X_test])				
				LD.data['Y_train'] = np.hstack([y_train, y_test])  
			else:
				LD.data['X_train'] = np.vstack([X_train, X_test])				
				LD.data['Y_train'] = np.hstack([y_train, y_test]) 
			

		except:
			try:
				LD.data['X_train'], LD.data['Y_train'] = shuffle(LD.data['X_train'], LD.data['Y_train'] , random_state=1)
			except:
				pass

		
		# one more data cut (shuffled, but still risky)
		try:
			if LD.info['is_sparse'] == 1:
				for acnt in range(5):
					if LD.data['X_train'].data.nbytes > 2.4*10**8:
						LD.data['X_train']  = LD.data['X_train'] [:len(LD.data['Y_train'])/2]
						LD.data['Y_train']  = LD.data['Y_train'] [:len(LD.data['Y_train'])/2]
			else:
				for acnt in range(5):
					if LD.data['X_train'] .nbytes > 2.4*10**8:
						LD.data['X_train']  = LD.data['X_train'] [:len(LD.data['Y_train'])/2]
						LD.data['Y_train']  = LD.data['Y_train'] [:len(LD.data['Y_train'])/2]
		except:
			pass
		

		# get yt in format of one label per column for CV
		try: 
			yt_raw = np.array(data_converter.convert_to_bin(LD.data['Y_train'], len(np.unique(LD.data['Y_train'])), False))
		except:
			yt_raw = LD.data['Y_train']
			
		
		#Strategy is that we will have N workers that will try prediction models listed in separate file (will be described later)
		# in shared data they will push CV score, and predictions for train, valid and test data
		# separate blender worker will use this data to create linear ensemble.
		
		
		# regardless of strategy, for competition purposes, it is good to have work in separate process, that can easily be killed
		# there are 2 events visible to all workers
		# a) stop writing - just to be sure that we don't kill process in the middle of writing predictions
		# b) start_growing - after this point, stop searching for best parameters, just build new trees or other similar strategy
		
		
		stop_writing_event = multiprocessing.Event()
		start_growing_event = multiprocessing.Event()
		
		manager = multiprocessing.Manager()
		shared_data = manager.Namespace()
		shared_data.LD = LD
		shared_data.yt_raw =yt_raw


		# Data will be preprocessed (several lines below)
		# beside N workers that will try to predict (later in code) on preprocessed data. 
		# We could have same code to do this on unprocessed data, and later try to ensemble it
		# For competition purposes, but beside that we could not afford many of workers 
		# (it could still called with same code with N=1), but we need slightly different heuristic for memory
		# this is only reason that this is separate and hardcoded code
		
		try:
			#we have hardcoded 5 dictionaries where worker will push data
			# worker is explained in own script
			shared_raw_data = manager.Namespace()
			shared_raw_data.raw_model = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			shared_raw_data.raw_model1 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			shared_raw_data.raw_model2 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			shared_raw_data.raw_model3 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			shared_raw_data.raw_model4 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
					
			
			b_raw = multiprocessing.Process(target=engine_serial.worker, args=([ shared_data, shared_raw_data, Lfold, Lstart, Ltime_budget]))
			b_raw.start()
		except:
			pass
		
		
		# Code is full of this kind of memory constraints, they are hardcoded/tuned for codalab VM
		# problem is that there is not swap file, and in case of memory overflow, all process will be killed
		# we will try some of preprocessing strategies  should be listed in engine_prep_models.py
		# all should be in sklearn format (should have fit and transform method). In file is examples of pipelines and custom model
		# that should be visible to engine_preprocess (this example is created as "logt" in this file).
		
		if psutil.phymem_usage()[2]  < 30:
			if LD.info['task'] != 'multilabel.classification':
				LD = engine_preprocess.preprocess(LD, Loutput_dir, Lstart, Ltime_budget, Lbasename, running_on_codalab, shared_data, Lfold, manager, yt_raw)
		
		
		#This is main part of strategy
		# We will create namespace for sharing data between workers
		shared_data = manager.Namespace()
		shared_data.LD = LD
		shared_data.yt_raw =yt_raw
		
		#In engine_models.py should be listed all models (sklearn format only) with addtional properties
		# model -  model instance format - see examples
		# blend_group - we will linear ensemble N best models, but don't wan't best of similar ones.
		#			  to avoid this from same group only best one will be ensembled
		# getter - updater - setter - after signal, getter is function that will read "interesting" parameter
		#                                        from model, updater will update it, and setter will change that parameter. This will repeat until end.
		#                                        example is "number of estimators  get 80, update 80+20, push 100, next time read 100, update 120 push 120..."
		# generator - parameters that will be changed in model in every iteration. 
		
		models = engine_models.get_models(shared_data)
		models_count = len(models)
		
		
		# We will create semaphore for workers (how many of them will be run simultaniously)
		# In workes is hardcoded that first 4 models doesn't depend on semaphore. This is just for competition strategy to force some models.
		# and is hardcoded for VM (some heuristic with multiprocessing.cpu_count() is better strategy)
		Lncpu = 4  #this 4 + 4 (first 4 fixed) +2 (serial and blender) = 10 
		semaphore = multiprocessing.Semaphore(Lncpu)
		
		#Creating N workers
		for Lnum in range(models_count): 
			exec("shared_data.worker"+str(Lnum) + ' = {"done":0, "score":0, ' +
				 '"preds_valid": None, "preds_test": None, ' +
				 '"model":' + models[Lnum] ["model"]+ ', ' +
				  '"blend_group": "%s", ' +  
				 '"getter": "%s", ' +  
				 '"updater": "%s", ' +  
				 '"setter": "%s", ' +  
				 '"generator": "%s" ' +  
				 '}')  % (models[Lnum] ['blend_group'],models[Lnum] ['getter'], models[Lnum] ['updater'], models[Lnum] ['setter'], models[Lnum] ['generator'])
		
		
		workers = [multiprocessing.Process(target=engine_worker.worker, args=([tr_no, shared_data, start_growing_event, Lfold, semaphore])) for tr_no in range(models_count)]
		for wr in workers:
			wr.start()
		
		b0 = multiprocessing.Process(target=engine_blender.blender, args=([ shared_data, shared_raw_data, models_count,  stop_writing_event, Loutput_dir, Lbasename, Lstart, Ltime_budget, Lfold]))
		b0.start()
		
		# we will use 25% of time for getter-updater-seter strategy
		grow_time = Ltime_budget /4
		almost_time =  Ltime_budget - (time.time() - Lstart) - 10 - grow_time
		
		time.sleep (almost_time) #wait for last 10 second and send signal to main worker to stop write prediction files
		start_growing_event.set()
		print "Grow signal sent", time.ctime(),  "time left",  Ltime_budget - (time.time() - Lstart)
		
		#if we are done we can save some time  (at most 1/4 time)
		while( (Ltime_budget - 40) >  (time.time() - Lstart)):
			for Lnum in range(models_count):
				try:
					all_done = 1
					exec("wr_data = shared_data.worker"+str(Lnum))
					if wr_data['done'] != 2:
						all_done = 0
				except:
					pass
			if all_done == 1:
				break
			time.sleep (1) 

		print "Stop signal sent", time.ctime(),  "time left",  Ltime_budget - (time.time() - Lstart)
		stop_writing_event.set()
		time.sleep (8) #almost 3 seconds left, should be OK
		
		#terminate leftover process
		try:
			for wr in workers:
				try:
					wr.terminate()
				except:
					pass
		except:
			pass
		
		try:
			b_raw.terminate()
		except:
			pass

		try:
			b0.terminate()
		except:
			pass
			
		print "Done", time.ctime(),  "time left",  Ltime_budget - (time.time() - Lstart)
	except Exception as e:
		print "exception in engine_automl", time.ctime(),  "left=",  Ltime_budget - (time.time() - Lstart), str(e)
		#terminate leftovers
		try:
			for wr in workers:
				try:
					wr.terminate()
				except:
					pass
		except:
			pass
		
		try:
			b_raw.terminate()
		except:
			pass

		try:
			b0.terminate()
		except:
			pass
		