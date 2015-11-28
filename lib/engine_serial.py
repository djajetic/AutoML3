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

				
def worker (sd, srd, Lfold, Lstart, Ltime_budget):
	try:
		#this is raw data, this data will soon disappear,
		#so this should be exec before any preprocessing
		
		
		Y_train = np.copy(sd.LD.data['Y_train'])
		X_train = np.copy(sd.LD.data['X_train'])
		X_valid = np.copy(sd.LD.data['X_valid'])
		X_test = np.copy(sd.LD.data['X_test'])
		
		split = int(len(Y_train)*0.5)
		Lnum = -1
		for model in [linear_model.LogisticRegression(random_state=101),
					ensemble.RandomForestClassifier(n_estimators=16,  max_depth=3, random_state=102),
					linear_model.LogisticRegression(random_state=103),
					ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=4, warm_start=False, random_state=104),
					ensemble.GradientBoostingClassifier(n_estimators=100, warm_start=False, learning_rate=0.1, random_state=105),
					]:
			Lnum += 1
			
			if Ltime_budget < 500 and (time.time() - Lstart) / Ltime_budget > 0.5 and Lnum > 0:
				break
				
			if (time.time() - Lstart) / Ltime_budget > 0.8 and Lnum > 0:
				break
				
			if psutil.phymem_usage()[2] > 80 and Lnum > 0:
				time.sleep(4)
				
			if psutil.phymem_usage()[2] > 90 and Lnum > 0:
				destroy_this = some_way #todo
		
			if Lfold < 2:
				if Lnum < 1:
					print "p1x", Lnum, X_train.shape, Y_train.shape, int(split/5)
					model.fit(X_train[:10], Y_train[:10])
					print "p11x", Lnum, int(split/5)
					model.fit(X_train[:int(split/5)], Y_train[:int(split/5)])
					print "p2x", Lnum
				else:
					model.fit(X_train[:split], Y_train[:split])
				if psutil.phymem_usage()[2] > 80 and Lnum > 0:
					time.sleep(4)
				print "s1", Lnum
				if psutil.phymem_usage()[2] > 90 and Lnum > 0:
					destroy_this = some_way #todo
				preds = model.predict_proba(X_train[split:])
				exec('CVscore = libscores.'+ sd.LD.info['metric']  + '(sd.yt_raw[split:], preds)')
				print "s2", Lnum, CVscore
			
			if Lfold==2:
				model2=copy.deepcopy(model)
				model2.fit(X_train[split:], Y_train[split:])
				preds2 = model2.predict_proba(X_train[:split])
				exec('CVscore2 = libscores.'+ sd.LD.info['metric']  + '(sd.yt_raw[:split], preds2)')
				
				CVscore =  (CVscore  + CVscore2)/2 
					
				del model2
			
			
				try:
					preds = np.vstack([preds2, preds])
				except:
					preds = np.hstack([preds2, preds])  #TODO -- proper check for 1D
			
			if Lfold == 4:
				splitcv = int(len(Y_train)/4)
				def cvfit_predict(xt, xv, yt, yv, model):					
					model.fit(xt, yt)
					predscv = model.predict_proba(xv)
					return  predscv
				
				cxt = X_train[splitcv:]
				cyt = Y_train[splitcv:]
				cxv = X_train[:splitcv]
				cyv = sd.yt_raw[:splitcv]
				cvp1 = cvfit_predict(cxt, cxv, cyt, cyv, model)
				exec('cvs1 = libscores.'+ sd.LD.info['metric']  + '(cyv, cvp1)')

				cxt = np.vstack([X_train[:splitcv], X_train[splitcv*2:]])
				cyt = np.hstack([Y_train[:splitcv], Y_train[splitcv*2:]])
				cxv = X_train[splitcv:splitcv*2]
				cyv = sd.yt_raw[splitcv:splitcv*2]
				cvp2 = cvfit_predict(cxt, cxv, cyt, cyv, model)
				exec('cvs2 = libscores.'+ sd.LD.info['metric']  + '(cyv, cvp2)')
				
				cxt = np.vstack([X_train[:splitcv*2], X_train[splitcv*3:]])
				cyt = np.hstack([Y_train[:splitcv*2], Y_train[splitcv*3:]])
				cxv = X_train[splitcv*2:splitcv*3]
				cyv = sd.yt_raw[splitcv*2:splitcv*3]
				cvp3 = cvfit_predict(cxt, cxv, cyt, cyv, model)
				exec('cvs3 = libscores.'+ sd.LD.info['metric']  + '(cyv, cvp3)')

				cxt = X_train[:splitcv*3]
				cyt = Y_train[:splitcv*3]
				cxv = X_train[splitcv*3:]
				cyv = sd.yt_raw[splitcv*3:]
				cvp4 = cvfit_predict(cxt, cxv, cyt, cyv, model)
				exec('cvs4 = libscores.'+ sd.LD.info['metric']  + '(cyv, cvp4)')

				CVscore =  (cvs1  + cvs2 + cvs3 + cvs4)/4
				
				try:
					cvp = np.vstack([cvp1, cvp2])
					cvp = np.vstack([cvp, cvp3])
					cvp = np.vstack([cvp, cvp4])
				except: #TODO proper 1D
					cvp = np.hstack([cvp1, cvp2])
					cvp = np.hstack([cvp, cvp3])
					cvp = np.hstack([cvp, cvp4])
				
				exec('CVscore2 = libscores.'+ sd.LD.info['metric']  + '(sd.yt_raw, cvp)')
				CVscore = (CVscore + CVscore2)/2
				
				preds = cvp
				
				del cvp1
				del cvp2
				del cvp3
			
			if psutil.phymem_usage()[2] > 80 and Lnum > 0:
					time.sleep(4)
				
			if psutil.phymem_usage()[2] > 90 and Lnum > 0:
					destroy_this = some_way #todo
					
			if Lnum > 0:
				model.fit(X_train, Y_train)
			
			if psutil.phymem_usage()[2] > 80 and Lnum > 0:
					time.sleep(4)
				
			if psutil.phymem_usage()[2] > 90 and Lnum > 0:
					destroy_this = some_way #todo
			
			preds_valid = model.predict_proba(X_valid)
			preds_test = model.predict_proba(X_test)

			if Lnum == 0:
				wd =  srd.raw_model
				wd['preds_valid'] = preds_valid
				wd['preds_test'] = preds_test
				wd['preds_2fld'] = preds
				wd['score'] = CVscore * 0.5 #not reilable
				wd['done'] = 1
				srd.raw_model = wd
				print "*********rmodel score = ", CVscore * 0.5
			if Lnum == 1:
				wd1 =  srd.raw_model1
				wd1['preds_valid'] = preds_valid
				wd1['preds_test'] = preds_test
				wd1['preds_2fld'] = preds
				wd1['score'] = CVscore
				wd1['done'] = 1
				srd.raw_model1 = wd1
				print "*********rmodel 1 score = ", CVscore
			if Lnum == 2:
				wd2 =  srd.raw_model2
				wd2['preds_valid'] = preds_valid
				wd2['preds_test'] = preds_test
				wd2['preds_2fld'] = preds
				wd2['score'] = CVscore
				wd2['done'] = 1
				srd.raw_model2 = wd2
				print "*********rmodel 2 score = ", CVscore
			if Lnum == 3:
				wd3 =  srd.raw_model3
				wd3['preds_valid'] = preds_valid
				wd3['preds_test'] = preds_test
				wd3['preds_2fld'] = preds
				wd3['score'] = CVscore
				wd3['done'] = 1
				srd.raw_model3 = wd3
				print "*********rmodel 3 score = ", CVscore
			if Lnum == 4:
				wd4 =  srd.raw_model4
				wd4['preds_valid'] = preds_valid
				wd4['preds_test'] = preds_test
				wd4['preds_2fld'] = preds
				wd4['score'] = CVscore
				wd4['done'] = 1
				srd.raw_model4 = wd4
				print "*********rmodel 4 score = ", CVscore
	except Exception as e:
			print 'exception in serial worker ' + '     ' +  str(e)
	

