# Preprocessing data
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
from scipy import sparse
import engine_worker
import engine_models
import engine_blender
import engine_prep_models
import engine_serial

#sample custom preprocessor
# in most cases will only go into exception (negative numbers)

class logt:
	def fit(self, x,y):
		pass
	def transform(self, x):
		return np.log(1+x)

		
def transformer(Lnum, sd, semaphore, sample_size):
	semaphore.acquire()
	try:
		while (1):
			if psutil.phymem_usage()[2] < 40:
				break
			time.sleep(2)
		exec("tr =  sd.transformer"+str(Lnum))
		
		model = tr['model']
		
		if len(sd.LD.data['Y_train']) > 100:
			split = int(len(sd.LD.data['Y_train'])*sample_size)
		else:
			split = int(len(sd.LD.data['Y_train'])/2)
		
		xt = sd.LD.data['X_train'][:split]
		yt = sd.LD.data['Y_train'][:split]
		xv = sd.LD.data['X_train'][split:]
		yv_raw = sd.yt_raw[split:] 
		
		if model != 'raw_data':
			model.fit(xt, yt)
			xt = model.transform(xt)
			xv = model.transform(xv)
		del model
		
		test_model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
		
		if psutil.phymem_usage()[2] > 50:
			time.sleep(10)
		if psutil.phymem_usage()[2] > 85:
			destroy = this_worker
			
		test_model.fit(xt, yt)
		
		preds = test_model.predict_proba(xv)
		
		exec('score = libscores.'+ sd.LD.info['metric']  + '(sd.yt_raw[split:], preds)')
		
		test_model = linear_model.LogisticRegression()
		
		if psutil.phymem_usage()[2] > 50:
			time.sleep(10)
		if psutil.phymem_usage()[2] > 85:
			destroy = this_worker
		
		test_model.fit(xt, yt)
		
		preds = test_model.predict_proba(xv)
		
		exec('score2 = libscores.'+ sd.LD.info['metric']  + '(sd.yt_raw[split:], preds)')
		
		score = (score + score2)/2
		
		
		print Lnum, score, tr['model']
		tr['score'] = score
		tr['done'] = 1
		exec("sd.transformer"+str(Lnum) + " = tr")
		semaphore.release()
	except Exception as e:
		print Lnum, "error", str(e), tr['model']
		tr['score'] = 0
		tr['done'] = 1
		semaphore.release()
		
	
def preprocessor(Lnum, sd):
	try:
		exec("tr =  sd.transformer"+str(Lnum))	
		model = tr['model']
		
		if model != 'raw_data':
			model.fit(sd.LD.data['X_train'], sd.LD.data['Y_train'])
			X_train = model.transform(sd.LD.data['X_train'])
			X_valid = model.transform(sd.LD.data['X_valid'])
			X_test = model.transform(sd.LD.data['X_test'])
			
			tr['X_train'] = X_train
			tr['X_valid'] = X_valid
			tr['X_test'] = X_test
			tr['done'] = 2
		else:
			tr['X_train'] = sd.LD.data['X_train']
			tr['X_valid'] = sd.LD.data['X_valid']
			tr['X_test'] = sd.LD.data['X_test']
			tr['done'] = 2
			
		exec("sd.transformer"+str(Lnum) + " = tr")
	except Exception as e:
		print 'exception in preprocessor process' + '     ' +  str(e)
				

def preprocess(LD, Loutput_dir, Lstart, Ltime_budget, Lbasename, running_on_codalab, shared_data, Lfold, manager, yt_raw):
	try:		
		if Ltime_budget < 500 and (time.time() - Lstart) / Ltime_budget > 0.1:
		    return LD

		split = int(len(LD.data['Y_train'])/2) 

		try:
			print shared_data.LD.info['is_sparse'] 
			
			if shared_data.LD.info['is_sparse'] == 1:
				return LD
			
			transformer_models = engine_prep_models.get_models(shared_data)

			transformers_count = len(transformer_models)
			
			Lsample_size  = 0.5 # 0.1 = 10%
			Lncpu = 6
			Ltransform_sample_time = 0.1 #0.05 = 5%
			Ltransform_time = 0.15 #0.15 = 15%
			
			semaphore = multiprocessing.Semaphore(Lncpu)
			
			#create atribute, they can be updated indepedently
			for Lnum in range(transformers_count): 
				exec("shared_data.transformer"+str(Lnum) + ' = {"done":0, "score":0, "preds_2fld2": None, ' +
					 '"X_valid": None, "X_test": None, "model":' + transformer_models[Lnum] + '}')
			
			#now we will try what can be done in n% of dedicated time with m% of data
			#if time_left_before_transform > Ltime_budget/5: #if we allready spent more then 20% of time we are on thin ice
			
			time_left_before_transform = Ltime_budget - (time.time() - Lstart)
			sysdate_before_transform = time.time()
			transform_dedicated_time = time_left_before_transform * Ltransform_sample_time
			
			transformers = [multiprocessing.Process(target=transformer, args=([tr_no, shared_data, semaphore, Lsample_size])) for tr_no in range(transformers_count)]
			for tr in transformers:
				tr.start()
			
			while (transform_dedicated_time >   time.time() - sysdate_before_transform):

				if Lfold == 1:
					transformers_done = 4 #to save time
				else:
					transformers_done = 2
				for tr_no in range(transformers_count):
					exec("tr_data =  shared_data.transformer"+str(tr_no))
					
					if tr_data['done'] == 0:
						transformers_done -= 1
			
				if transformers_done > 0:
					break
				time.sleep (1) 
			
			try:
				del semaphore 
			except:
				pass
			
			for tr in transformers:
				try:
					tr.terminate()
				except:
					pass

					
			transformers_data = []
			for tr_no in range(transformers_count):
				exec("tr_data =  shared_data.transformer"+str(tr_no))
				if tr_data['done'] == 1:
					transformers_data.append(tr_data)
			
			transformers_data = sorted(transformers_data, key=itemgetter('score'), reverse=True)
			
			#we'll take 4 just in case, if some don't finish on time
			transformers_data = transformers_data[:4]
			
			print "transformers data", transformers_data[:1]
			if len(transformers_data) > 3:
				
				del shared_data
				
				shared_data = manager.Namespace()
				shared_data.LD = LD
				shared_data.yt_raw =yt_raw
				
				
				time_left_before_transform = Ltime_budget - (time.time() - Lstart)
				sysdate_before_transform = time.time()
				transform_dedicated_time = time_left_before_transform * Ltransform_time
				
				transformers_count = len(transformers_data)
				
				for Lnum in range(transformers_count): 
					exec("shared_data.transformer"+str(Lnum) + ' = transformers_data[Lnum]')
				
				transformers = [multiprocessing.Process(target=preprocessor, args=([tr_no, shared_data])) for tr_no in range(transformers_count)]
				for tr in transformers:
					tr.start()
				
				while (transform_dedicated_time >   time.time() - sysdate_before_transform):
					tr_data =  shared_data.transformer0
					if tr_data['done'] == 2:
						break
					time.sleep (1) 

				#terminate leftovers
				for tr in transformers:
					try:
						tr.terminate()
					except:
						pass
						
				
				#if time is over (or first one is done), use this data
				for tr_no in range(transformers_count):
					exec("tr_data =  shared_data.transformer"+str(tr_no))
					if tr_data['done'] == 2:
						LD.data['X_train'] = tr_data['X_train'] 
						LD.data['X_valid'] = tr_data['X_valid'] 
						LD.data['X_test'] = tr_data['X_test'] 
						break 
						
			del shared_data
			del tr_data
			return LD
		except Exception as e:
			print "exception in engine_preprocess", time.ctime(),  "left=",  Ltime_budget - (time.time() - Lstart), str(e)
			return LD
		
	except Exception as e:
		print "exception in engine_preprocess2", time.ctime(),  "left=",  Ltime_budget - (time.time() - Lstart), str(e)
		return LD
