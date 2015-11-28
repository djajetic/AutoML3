# List of models that will be tested
# Damir Jajetic, 2015
def get_models(sd):

#this models that will be cross validated on test data


	if sd.LD.info['is_sparse'] == 1:
		models = [
			{"model": 'naive_bayes.BernoulliNB(alpha=0.1)',
			  "blend_group":   "NB",  
			    "getter":   "Lalpha = model_last.get_params()['alpha']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "alpha=0.01 @@ " \
						  "alpha=0.1 @@ " \
						  "alpha=0.4 @@ " \
						  "alpha=1 @@ " \
						  "alpha=0.001 @@ " \
						  "alpha=8 @@ " \
						  "alpha=0.02 @@ " 
			},
			{"model": 'linear_model.LogisticRegression(C=0.1, random_state=Lnum)',
			    "blend_group":   "LC",  
			    "getter":   "Lalpha = model_last.get_params()['alpha']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "C=0.01 @@ " \
						  "C=0.1 @@ " \
						  "C=0.4 @@ " \
						  "C=1 @@ " \
						  "C=0.001 @@ " \
						  "C=8 @@ " \
						  "C=0.02 @@ " 
						  "C=20 @@ " 
			},
			{"model": 'linear_model.LogisticRegression(penalty="l1", C=0.1, random_state=Lnum)',
			"blend_group":   "LC",  
			    "getter":   "Lalpha = model_last.get_params()['alpha']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "C=0.01 @@ " \
						  "C=0.1 @@ " \
						  "C=0.4 @@ " \
						  "C=1 @@ " \
						  "C=0.001 @@ " \
						  "C=8 @@ " \
						  "C=0.02 @@ " 
						  "C=20 @@ " 
			},
			{"model": 'neighbors.KNeighborsClassifier(n_neighbors=4)',
			"blend_group":   "KNN",  
			    "getter":   "Lalpha = model_last.get_params()['alpha']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "n_neighbors=4 @@ " \
						  "n_neighbors=2 @@ " \
						  "n_neighbors=8 @@ " \
						  "n_neighbors=3 @@ " \
			}
			]
	else:
		models = [
			{"model": 'linear_model.LogisticRegression(random_state=1)',
			"blend_group":   "LC",  
			    "getter":   "Lestimators = model_last.get_params()['penalty']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "penalty='l2', dual=False, C=1.0 @@ " \
						  "penalty='l1', dual=False, C=1.0 @@ " \
						  "penalty='l2', dual=False, C=2.0 @@ " \
						  "penalty='l2', dual=False, C=0.5 @@ " \
						  "penalty='l2', dual=True, C=1.0 @@ " \
						  "penalty='l2', dual=True, C=0.5 @@ " \
						  "penalty='l2', dual=False, C=4.0 @@ " \
						  "penalty='l2', dual=False, C=8.0 @@ " 
						  "penalty='l2', dual=False, C=1.0 @@ " \
						  "penalty='l1', dual=False, C=1.0 @@ " \
						  "penalty='l2', dual=False, C=2.0 @@ " \
						  "penalty='l2', dual=False, C=0.5 @@ " \
						  "penalty='l2', dual=True, C=1.0 @@ " \
						  "penalty='l2', dual=True, C=0.5 @@ " \
						  "penalty='l2', dual=False, C=4.0 @@ " \
						  "penalty='l2', dual=False, C=8.0 @@ " 
			},
			{"model": 'naive_bayes.GaussianNB()',
			"blend_group":   "NB",  
			    "getter":   "Lestimators = model_last.get_params()",
			    "updater":   "tries_left = 0",
			     "setter":   "",
			     "generator":  "" 
			},
			{"model": 'f16.RandomForestClassifier(n_estimators=60, random_state=Lnum, n_jobs=1)',
			"blend_group":   "RDT1",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators, warm_start = True",
			    "generator": 
						"max_depth=2 @@" \
						"max_depth=6 @@" \
						"min_samples_split=8 @@" \
						"min_samples_split=4 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_features=0.8 @@" \
						"max_depth=None @@" \
						"min_samples_leaf=2 @@" \
						"min_samples_leaf=4 @@" \
						"min_samples_leaf=8 @@" \
						"max_features=0.6 @@" \
						"max_depth=2 @@" \
						"max_depth=8 @@" \
						"min_samples_leaf=6 @@" \
						"max_depth=4 @@" \
						"min_samples_leaf=16 @@" \
						"max_depth=6 @@" \
						"max_depth=8 @@" \
						"criterion='entropy' @@" \
						"min_samples_leaf=32 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_depth=None @@" \
						"min_samples_split=2 @@" \
						"min_samples_leaf=1 @@" \
						"max_features=0.9 @@" \
						"max_features=0.95 @@" \
							"max_depth=2 @@" \
						"max_depth=6 @@" \
						"min_samples_split=8 @@" \
						"min_samples_split=4 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_features=0.8 @@" \
						"max_depth=None @@" \
						"min_samples_leaf=2 @@" \
						"min_samples_leaf=4 @@" \
						"min_samples_leaf=8 @@" \
						"max_features=0.6 @@" \
						"max_depth=2 @@" \
						"max_depth=8 @@" \
						"min_samples_leaf=6 @@" \
						"max_depth=4 @@" \
						"min_samples_leaf=16 @@" \
						"max_depth=6 @@" \
						"max_depth=8 @@" \
						"criterion='entropy' @@" \
						"min_samples_leaf=32 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_depth=None @@" \
						"min_samples_split=2 @@" \
						"min_samples_leaf=1 @@" \
						"max_features=0.9 @@" \
						"max_features=0.95 @@" \
			},
			{"model": 'ensemble.GradientBoostingClassifier(n_estimators=60, warm_start=True, random_state=Lnum)',
			"blend_group":   "GB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 10",
			     "setter":   "n_estimators = Lestimators , warm_start = True",
			     "generator": ""
			},

			{"model": 'f16.RandomForestClassifier(n_estimators=60, warm_start = True, random_state=Lnum, n_jobs=1)',
			"blend_group":   "RDT3",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": ""
			},
			{"model": 'ensemble.GradientBoostingClassifier(n_estimators=60, warm_start=False, learning_rate=0.2, random_state=Lnum)',
			   "blend_group":   "GB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			     "setter":   "n_estimators = Lestimators, warm_start = True",
			     "generator":  "max_depth=2 @@" \
						"max_depth=6 @@" \
						"min_samples_split=8 @@" \
						"min_samples_split=4 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_features=0.8 @@" \
						"max_depth=None @@" \
						"min_samples_leaf=2 @@" \
						"min_samples_leaf=4 @@" \
						"min_samples_leaf=8 @@" \
						"max_features=0.6 @@" \
						"max_depth=2 @@" \
						"max_depth=8 @@" \
						"min_samples_leaf=6 @@" \
						"max_depth=4 @@" \
						"min_samples_leaf=16 @@" \
						"max_depth=6 @@" \
						"max_depth=8 @@" \
						"learning_rate=0.3 @@" \
						"subsample=0.95 @@" \
						"min_samples_leaf=32 @@" \
						"max_features='auto' @@" \
						"max_depth=None @@" \
						"subsample=0.9 @@" \
						"max_features=None @@" \
						"min_samples_split=2 @@" \
						"subsample=0.8 @@" \
						"min_samples_leaf=1 @@" \
						"max_features=0.9 @@" \
						"max_features=0.95 @@" \
						"learning_rate=0.1 @@" \
						"learning_rate=0.05 @@" \
						"learning_rate=0.02 @@" \
						"max_depth=2 @@" \
						"max_depth=6 @@" \
						"min_samples_split=8 @@" \
						"min_samples_split=4 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_features=0.8 @@" \
						"max_depth=None @@" \
						"min_samples_leaf=2 @@" \
						"min_samples_leaf=4 @@" \
						"min_samples_leaf=8 @@" \
						"max_features=0.6 @@" \
						"max_depth=2 @@" \
						"max_depth=8 @@" \
						"min_samples_leaf=6 @@" \
						"max_depth=4 @@" \
						"min_samples_leaf=16 @@" \
						"max_depth=6 @@" \
						"max_depth=8 @@" \
						"learning_rate=0.3 @@" \
						"subsample=0.95 @@" \
						"min_samples_leaf=32 @@" \
						"max_features='auto' @@" \
						"max_depth=None @@" \
						"subsample=0.9 @@" \
						"max_features=None @@" \
						"min_samples_split=2 @@" \
						"subsample=0.8 @@" \
						"min_samples_leaf=1 @@" \
						"max_features=0.9 @@" \
						"max_features=0.95 @@" \
						"learning_rate=0.1 @@" \
						"learning_rate=0.05 @@" \
						"learning_rate=0.02 @@" \
			},
						{"model": 'f16.ExtraTreesClassifier(n_estimators=60, random_state=Lnum, n_jobs=1)',
			  "blend_group":   "RDT2",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators, warm_start = True",
			    "generator": 
						"max_depth=2 @@" \
						"max_depth=6 @@" \
						"min_samples_split=8 @@" \
						"min_samples_split=4 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_features=0.8 @@" \
						"max_depth=None @@" \
						"min_samples_leaf=2 @@" \
						"min_samples_leaf=4 @@" \
						"min_samples_leaf=8 @@" \
						"max_features=0.6 @@" \
						"max_depth=2 @@" \
						"max_depth=8 @@" \
						"min_samples_leaf=6 @@" \
						"max_depth=4 @@" \
						"min_samples_leaf=16 @@" \
						"max_depth=6 @@" \
						"max_depth=8 @@" \
						"criterion='entropy' @@" \
						"min_samples_leaf=32 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_depth=None @@" \
						"min_samples_split=2 @@" \
						"min_samples_leaf=1 @@" \
						"max_features=0.9 @@" \
						"max_features=0.95 @@" \
							"max_depth=2 @@" \
						"max_depth=6 @@" \
						"min_samples_split=8 @@" \
						"min_samples_split=4 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_features=0.8 @@" \
						"max_depth=None @@" \
						"min_samples_leaf=2 @@" \
						"min_samples_leaf=4 @@" \
						"min_samples_leaf=8 @@" \
						"max_features=0.6 @@" \
						"max_depth=2 @@" \
						"max_depth=8 @@" \
						"min_samples_leaf=6 @@" \
						"max_depth=4 @@" \
						"min_samples_leaf=16 @@" \
						"max_depth=6 @@" \
						"max_depth=8 @@" \
						"criterion='entropy' @@" \
						"min_samples_leaf=32 @@" \
						"max_features='auto' @@" \
						"max_features=None @@" \
						"max_depth=None @@" \
						"min_samples_split=2 @@" \
						"min_samples_leaf=1 @@" \
						"max_features=0.9 @@" \
						"max_features=0.95 @@" \
			},
			{"model": 'ensemble.AdaBoostClassifier(n_estimators=60, random_state=Lnum)',
			   "blend_group":   "AB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": ""
			},
			{"model": 'ensemble.AdaBoostClassifier(base_estimator=f16.RandomForestClassifier(n_estimators=20), n_estimators=60, random_state=Lnum)',
			"blend_group":   "AB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": "base_estimator=f16.RandomForestClassifier(n_estimators=5) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=10) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=20) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=40) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=60) @@" \
			},
			{"model": 'ensemble.AdaBoostClassifier(base_estimator=f16.RandomForestClassifier(n_estimators=20, max_depth=4), n_estimators=60, random_state=Lnum)',
			"blend_group":   "AB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": "base_estimator=f16.RandomForestClassifier(n_estimators=5, max_depth=4) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=10, max_depth=4) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=20, max_depth=4) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=40, max_depth=4) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=60, max_depth=4) @@" \
			},
			{"model": 'ensemble.AdaBoostClassifier(base_estimator=f16.RandomForestClassifier(n_estimators=20, max_depth=6), n_estimators=60, random_state=Lnum)',
			"blend_group":   "AB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": "base_estimator=f16.RandomForestClassifier(n_estimators=5, max_depth=6) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=10, max_depth=6) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=20, max_depth=6) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=40, max_depth=6) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=60, max_depth=6) @@" \
			},
			{"model": 'ensemble.AdaBoostClassifier(base_estimator=f16.RandomForestClassifier(n_estimators=20, min_samples_leaf=6), n_estimators=60, random_state=Lnum)',
			"blend_group":   "AB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator":  "base_estimator=f16.RandomForestClassifier(n_estimators=5, min_samples_leaf=6) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=10, min_samples_leaf=6) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=20, min_samples_leaf=6) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=40, min_samples_leaf=6) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=60, min_samples_leaf=6) @@" \
			},
			{"model": 'ensemble.AdaBoostClassifier(base_estimator=f16.RandomForestClassifier(n_estimators=20, max_features=0.8), n_estimators=60, random_state=Lnum)',
			"blend_group":   "AB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": "base_estimator=f16.RandomForestClassifier(n_estimators=5, max_features=0.8) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=10, mmax_features=0.8) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=20, max_features=0.8) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=40, max_features=0.8) @@" \
						"base_estimator=f16.RandomForestClassifier(n_estimators=60, max_features=0.8) @@" \
			},
			{"model": 'ensemble.AdaBoostClassifier(base_estimator=ensemble.ExtraTreesClassifier(n_estimators=20), n_estimators=60, random_state=Lnum)',
			"blend_group":   "AB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": "ensemble.ExtraTreesClassifier(n_estimators=5) @@" \
						"ensemble.ExtraTreesClassifier(n_estimators=10) @@" \
						"ensemble.ExtraTreesClassifier(n_estimators=20) @@" \
						"ensemble.ExtraTreesClassifier(n_estimators=40) @@" \
						"ensemble.ExtraTreesClassifier(n_estimators=60) @@" \
						
			},
			{"model": 'ensemble.AdaBoostClassifier(base_estimator=ensemble.RandomForestClassifier(n_estimators=60), n_estimators=60, random_state=Lnum)',
			"blend_group":   "AB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": "ensemble.RandomForestClassifier(n_estimators=5) @@" \
						"ensemble.RandomForestClassifier(n_estimators=10) @@" \
						"ensemble.RandomForestClassifier(n_estimators=20) @@" \
						"ensemble.RandomForestClassifier(n_estimators=40) @@" \
						"ensemble.RandomForestClassifier(n_estimators=60) @@" \
			},


			{"model": 'neighbors.KNeighborsClassifier()',
			"blend_group":   "KNN",  
			    "getter":   "Lestimators = model_last.get_params()['n_neighbors']",
			    "updater":   "tries_left = 0",
			     "setter":   "",
			     "generator":  "n_neighbors=3 @@ " \
						   "n_neighbors=5 @@ " \
						  "n_neighbors= 8 @@ "  \
						   "n_neighbors=5  weights='distance' " 
						 
			},
			{"model": 'Pipeline([("pre", preprocessing.StandardScaler()),("svc", svm.SVC(probability=True, random_state=Lnum, C=1.0))])',
			"blend_group":   "SVM",  
			    "getter":  "Lc = model_last.get_params()['svc__C']",
			    "updater":  "Lc *= 0.8",
			     "setter":   "svc__C=Lc",
			     "generator": "svc__kernel= 'linear', svc__C=1 @@" \
						 "svc__kernel= 'linear', svc__C=0.4 @@" \
						  "svc__kernel= 'linear', svc__C=2 @@" \
						  "svc__kernel= 'linear', svc__C=4 @@" \
						  "svc__kernel= 'poly', svc__degree=2 @@" \
						  " svc__degree=3 @@" \
						  "svc__kernel= 'rbf' @@" \
	"					svc__degree=2, svc__C=1 @@" \
						" svc__C=1 @@" \
						" svc__C=0.4 @@" \
						" svc__C=24 @@" \
						" svc__C=2 @@" \
						" svc__C=8 @@" \
						" svc__C=0.01 @@" \
						" svc__gamma=0.1 @@" \
						" svc__gamma=0.4 @@" \
						" svc__coef0=1 @@" \
			},
			{"model": 'svm.SVC(probability=True, C=1.0)',
			"blend_group":   "SVM",  
			   "getter":  "Lc = model_last.get_params()['C']",
			    "updater":  "Lc *= 0.8",
			     "setter":   "C= Lc",
			     "generator":  "C=1 @@" \
						   "C=0.1 @@" \
						    "C=0.4 @@" \
						     "C=4 @@" \
						     "class_weight='auto' @@" \
						     "gamma=0.1 @@" \
						     "gamma=0.4 @@" \
						     "C=8 @@" \
						     "coef0=1 @@" \
						     "C=0.04 @@" \
						     "C=0.01 @@" \
						      
			}
			]
		
	return models
