#Damir Jajetic, 2015
import copy
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
if sklearn.__version__ >= 0.16:
	from sklearn import ensemble as f16
else:
	try:
		import forest16 as f16 #this is backport of sklearn tree 0.16 version  for 0.15 sklearn created by aad_freiburg
	except:
		from sklearn import ensemble as f16

def worker (Lnum, sd, start_growing, Lfold, semaphore):
 try:	
	
	# this generator will yield parameters that model should try
	# parameters are defined in engine_models.py
	def f_parameter_generator(parameters_list):
		for parameters in parameters_list:
			yield parameters
		yield "done"
	
	#this is just tuned for codalab VM, for first 4 models don't use semaphore
	if Lnum > 4:
		time.sleep(Lnum)
	
	if Lnum > 4:
		sema = 1
		semaphore.acquire()

	#models are not set in parameters, but every worker pull data belonging his parameter number
	#this way most errors in model setup will generate problems (in main process), wrrors in worker is handled by "pass"
	exec("wd =  sd.worker"+str(Lnum))
	split = int(len(sd.LD.data['Y_train'])*0.5)
	
	#Sometimes we will need to roll back to last or best model
	#this will increase memmory consumptions, pickle creation should be considered as alternative
	model = wd['model']
	model_all = copy.deepcopy(wd['model']) 
	model_last = copy.deepcopy(wd['model']) 
	model_best = copy.deepcopy(wd['model']) 
	best_CVscore = 0
		
		
	gen_list = wd['generator'].split('@@')
	parameter_generator = f_parameter_generator(gen_list)
		
	use_generator = 1
	
	tries_left = 10
	while (tries_left > 0):

		if psutil.phymem_usage()[2] > 60:
			time.sleep(2)
			continue 
	
		if Lnum > 4:
			if sema == 1:
				sema = 0
			else:
				semaphore.acquire()
		
		try:
			if start_growing.is_set() == True:
				print Lnum, "start growing set"	
				use_generator = 0
			if use_generator == 1:
				setter = parameter_generator.next()
				if setter == "done":
					print Lnum, "generator done", time.ctime()	
					use_generator = 0

					model_last = copy.deepcopy(model_best)
					model = copy.deepcopy(model_best)
					model_all = copy.deepcopy(model_best)
					
					exec(wd['getter'])
					exec(wd['updater'])
					setter = wd['setter']
			else:
				model = copy.deepcopy(model_best) 
				exec(wd['getter'])
				exec(wd['updater'])
				setter = wd['setter']


			if psutil.phymem_usage()[2] > 70:
				time.sleep(1)
			if psutil.phymem_usage()[2] > 85:
				destroy = this_worker  #will go to exception	
				
			exec("model.set_params(" + setter + ")")
			
			
			model.fit(sd.LD.data['X_train'][:split], sd.LD.data['Y_train'][:split])
			
			if psutil.phymem_usage()[2] > 70:
				time.sleep(1)
			if psutil.phymem_usage()[2] > 85:
				destroy = this_worker
			
			preds = model.predict_proba(sd.LD.data['X_train'][split:])
			
			if psutil.phymem_usage()[2] > 60:
				time.sleep(4)
			if psutil.phymem_usage()[2] > 85:
				destroy = this_worker


			if sd.LD.info['task'] == 'multilabel.classification':
				preds = np.array(preds)
				preds = preds[:, :, 1]
				preds.reshape(preds.shape[0],preds.shape[1])
				preds = preds.T

			exec('CVscore = libscores.'+ sd.LD.info['metric']  + '(sd.yt_raw[split:], preds)')

			
			if Lfold == 2:
				model2=copy.deepcopy(model)
				try:
					model2.set_params(warm_start=False) # :)
				except:
					pass
				model2.fit(sd.LD.data['X_train'][split:], sd.LD.data['Y_train'][split:])
				preds2 = model2.predict_proba(sd.LD.data['X_train'][:split])
				exec('CVscore2 = libscores.'+ sd.LD.info['metric']  + '(sd.yt_raw[:split], preds2)')
				
				CVscore =  (CVscore  + CVscore2)/2 
				
				del model2
				
			#in original version here was 4 fold CV version that was unused (when running on codalab this parameter is set to Lfold=1)
			#but just for readability this is removed (implementation can still be found in codalab public submissions)
			
			model_last = copy.deepcopy(model) 
			
			if CVscore > best_CVscore:
				if psutil.phymem_usage()[2] > 70:
					time.sleep(1)				
				if psutil.phymem_usage()[2] > 85:
					destroy = this_worker
					
				model_best = copy.deepcopy(model)
				
				exec("model_all.set_params(" + setter + ")") #because of warm start
				
				if psutil.phymem_usage()[2] > 70:
					time.sleep(1)				
				if psutil.phymem_usage()[2] > 85:
					destroy = this_worker
					
				time.sleep(0.05)
				best_CVscore = CVscore
			
				model_all.fit(sd.LD.data['X_train'], sd.LD.data['Y_train']) 
				
				if psutil.phymem_usage()[2] > 60:
					time.sleep(4)				
				if psutil.phymem_usage()[2] > 85:
					destroy = this_worker
				
				preds_valid = model_all.predict_proba(sd.LD.data['X_valid'])
				preds_test = model_all.predict_proba(sd.LD.data['X_test'])
				
				
				if sd.LD.info['task'] == 'multilabel.classification':
					preds_valid = np.array(preds_valid)
					preds_valid = preds_valid[:, :, 1]
					preds_valid.reshape(preds_valid.shape[0],preds_valid.shape[1])
					preds_valid = preds_valid.T
					
					preds_test = np.array(preds_test)
					preds_test = preds_test[:, :, 1]
					preds_test.reshape(preds_test.shape[0],preds_test.shape[1])
					preds_test = preds_test.T


				wd['score'] = CVscore				
				wd['preds_2fld'] = preds
				
				if Lfold == 2:
					try:
						wd['preds_2fld'] = np.vstack([preds2, preds])
					except:
						wd['preds_2fld'] = np.hstack([preds2, preds])  #TODO -- proper check for 1D
						
				
				wd['preds_valid'] = preds_valid
				wd['preds_test'] = preds_test
				
				if wd['done'] == 0: wd['done'] = 1
				exec("sd.worker"+str(Lnum) + " =  wd")
				
				if use_generator == 0:
					tries_left = 10
			else:
				if use_generator == 0:
					tries_left -= 1  #try 10 times after  best model
			if Lnum > 4:
				semaphore.release()
		except Exception as e:
			try:
				print 'exception in worker ' + '     ' +  str(e)
				print "error model = ", wd['model']
			except:
				pass
			try:
				if Lnum > 4:
					semaphore.release()
				break
			except:
				pass

			#continue
	
	#pull push all data
	exec("wd =  sd.worker"+str(Lnum))
	if wd['done'] == 1: wd['done'] = 2
	exec("sd.worker"+str(Lnum) + " =  wd")
	print Lnum, "all done", time.ctime(), model
 except Exception as e:
	print 'out exception in worker ' + '     ' +  str(e)
	try:
		print "error model = ", wd['model']
	except:
		pass
	try:
		if Lnum > 4:
			semaphore.release()
	except:
		pass

