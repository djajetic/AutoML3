# List of models for data preprocessing
#Damir Jajetic, 2015
def get_models(sd):
	models = [
			'"raw_data"',
			'logt()',
			'preprocessing.StandardScaler()',
			'preprocessing.Normalizer()',
			'ensemble.RandomForestClassifier(n_estimators=200,  max_depth=8,random_state=0)',
			'preprocessing.MinMaxScaler()', 
			'Pipeline([("rf", ensemble.RandomForestClassifier(n_estimators=200, max_depth=8,random_state=Lnum)),("nor", preprocessing.Normalizer())])',
			'Pipeline([("rf", ensemble.RandomForestClassifier(n_estimators=200, max_depth=8,random_state=Lnum)),("nor", preprocessing.StandardScaler())])',
			'Pipeline([("rf", ensemble.RandomForestClassifier(n_estimators=200, max_depth=8,random_state=Lnum)),("pca", decomposition.PCA())])',
			'Pipeline([("rf", ensemble.RandomForestClassifier(n_estimators=200, max_depth=8,random_state=Lnum)),("pca", decomposition.PCA()), ("nor", preprocessing.StandardScaler())])',
			#'preprocessing.PolynomialFeatures()', check mem
			'decomposition.PCA()',
			'Pipeline([("mm", preprocessing.StandardScaler()), ("bin", preprocessing.Binarizer())])',
			'random_projection.GaussianRandomProjection()',
			'random_projection.SparseRandomProjection()',
			'decomposition.PCA(n_components="mle")',
			'Pipeline([("mm", preprocessing.MinMaxScaler()), ("nn", neural_network.BernoulliRBM(random_state=0))])',
			'kernel_approximation.AdditiveChi2Sampler()',
			'decomposition.RandomizedPCA()',
			'manifold.Isomap()',
			'cross_decomposition.PLSRegression()',
			'cross_decomposition.CCA()',
			'feature_selection.VarianceThreshold(threshold=0.05)',
			'feature_selection.VarianceThreshold(threshold=0.1)',
			'feature_selection.VarianceThreshold(threshold=0.2)',
			'decomposition.FastICA()',
			'decomposition.KernelPCA(kernel="rbf")',
			'decomposition.KernelPCA(kernel="poly")',
			'decomposition.KernelPCA(kernel="sigmoid")',
			'decomposition.KernelPCA(kernel="cosine")',
			'Pipeline([("rf", ensemble.RandomForestClassifier(n_estimators=200, max_depth=8, random_state=Lnum)),("rbf", decomposition.KernelPCA(kernel="rbf")),("sta", preprocessing.StandardScaler()),("nor", preprocessing.Normalizer())])',
			'Pipeline([("rf", ensemble.RandomForestClassifier(n_estimators=200, max_depth=8, random_state=Lnum)),("sta", preprocessing.StandardScaler()),("nor", preprocessing.StandardScaler())])',
			'Pipeline([("rf", ensemble.RandomForestClassifier(n_estimators=200, max_depth=8, random_state=Lnum)),("log", logt()),("sta", preprocessing.StandardScaler()),("nor", preprocessing.StandardScaler())])',
			'Pipeline([("rf", ensemble.RandomForestClassifier(n_estimators=200, max_depth=8, random_state=Lnum)),("var", feature_selection.VarianceThreshold(threshold=0.1)), ("nor", preprocessing.StandardScaler())])',
			'decomposition.MiniBatchSparsePCA()']
	
	return models