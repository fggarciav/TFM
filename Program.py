import sys
# Import scikit-learn library functions
from sklearn import feature_extraction,decomposition,impute,preprocessing,manifold,neighbors
# Import MatPlot and NumPy libraries
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import json
import hdbscan
import time
import seaborn as sns

def mysql_connection(connection_file):

    # Read json connection details
    with open(connection_file) as f:
        conn = json.loads(f.read())
        f.close()

    # Ensure your connection variables were setup
    if conn:
       username = (conn['username'])
       password = (conn['password'])
       database = (conn['database'])
       host = (conn['host'])
       port = (conn['port'])

       # Create  engine connection to MySQL Database
       conn_string = 'mysql+pymysql://' + username + ':' + password + '@' + host + ':' + port + '/' + database
       engine = create_engine(conn_string)

       return engine

def preprocessing_dataset(attributes):
    # Drop NumTransaccion column
    attributes = attributes.drop("NumTransaccion", axis=1)

    # Impute missing Dataframe values
    attributes = attributes.fillna(value="0")
    #imp = impute.SimpleImputer(missing_values=None, strategy="constant", fill_value="None")
    #attributes = imp.fit_transform(attributes)

    # Convert string dates to timestamp format
    attributes['FechaTransaccion'] = pd.to_datetime(attributes['FechaTransaccion'], format='%Y%m%d')
    attributes['FechaTransaccion'] = (attributes['FechaTransaccion'] - pd.Timestamp("2000-01-01")) // pd.Timedelta('1d')

    # Transform 'IdUsuario' values to sparse matrix with feature hashing (n_features = 4.096 < unique(IdUsuario))
    enc = feature_extraction.FeatureHasher(n_features=2**12,input_type='string')
    # Transform feature and retorn sparse matrix
    sp = enc.fit_transform(attributes['IdUsuario'])
    # Convert sparse matrix to DataFrame format
    df = pd.DataFrame(sp.toarray())
    # Concatenate feature transform with original dataset and remove original attribute column
    idx_attributes = attributes.index.tolist()
    attributes = pd.concat([attributes.reset_index(drop=True),df.reset_index(drop=True)],axis=1)
    # Setting old index to continue using the same row identification
    attributes.index = idx_attributes
    # Drop column 'IdUsuario' transformed
    attributes.drop(['IdUsuario'], axis=1, inplace=True)

    # Transform string values of 'Valor' column to numerical values
    attributes['Valor'] = pd.to_numeric(attributes['Valor'],errors='coerce')
    # Replace non numèric values (previously converted in NaN values) to 0 value.
    attributes['Valor'] = attributes['Valor'].replace(np.nan, 0, regex=True)
    attributes['Valor'] = attributes['Valor'].astype(int)

    # # Convert DataFrame to numpy array
    dataset = attributes.to_numpy()

    # Return dataset in array format
    return dataset

def data_standardization(attributes):

    # Standardize dataset
    dataset = preprocessing.scale(attributes)
    return dataset

def pca_decomposition(attributes):

    # PCA Decomposition fit with 95% of accumulated explained variance
    pca = decomposition.PCA(.95)
    pca.fit(attributes)

    # Show PCA trained results
    print('Principal components requiered:',pca.n_components_)
    print('Explained variance represented (%):',sum(pca.explained_variance_ratio_))

    # Represent accumulated explained variance curve.
    plot.title('PCA Accumulated explained variance')
    plot.plot(np.cumsum(pca.explained_variance_ratio_))
    plot.xlabel('n_components')
    plot.ylabel('Accumulated explained variance (%)')
    plot.show()

    # Dataset decomposition
    dataset = pca.transform(attributes)

    return dataset

def svd_decomposition(attributes):

    # SVD decomposition with mínimal a 95% of accumulated explained variance
    svd = decomposition.TruncatedSVD(37, n_iter=5)
    svd.fit(attributes)

    # Show SVD trained results
    print('Singular vector decomposition components used:',svd.n_components)
    print('Explained variance represented (%):',sum(svd.explained_variance_ratio_))

    # Represent accumulated explained variance curve.
    plot.title('SVD Accumulated explained variance')
    plot.plot(np.cumsum(svd.explained_variance_ratio_))
    plot.xlabel('n_components')
    plot.ylabel('Accumulated explained variance (%)')
    plot.show()

    # Dataset decomposition
    dataset = svd.transform(attributes)

    return dataset

def mds_reduction2d(attributes):
    # Multidimensional Scaling in 2-dimensions
    mds = manifold.MDS(n_components=2, n_init=4, n_jobs=-1)
    attributes = mds.fit_transform(attributes)

    return attributes

def hdbscan_clustering(attributes):
    min_cluster_size = 2
    min_samples = None
    hdb_t1 = time.time()
    # Hierarchical Density-Based Spatial Clustering of Applications with Noise
    clf = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples, prediction_data=True).fit(attributes)
    hdb_labels = clf.labels_
    hdb_elapsed_time = time.time() - hdb_t1
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

    print('\n\n++++ HDBSCAN Results ++++')
    print('Estimated number of clusters: %d' % n_clusters_hdb_)
    print('Elapsed time to cluster: %.4f s' % hdb_elapsed_time)
    threshold = pd.Series(clf.outlier_scores_).quantile(0.99)
    outliers = np.where(clf.outlier_scores_ > threshold)[0]
    print('Estimated number of outliers: ', len(outliers))

    # # Draw distribution plot
    sns.distplot(clf.outlier_scores_[np.isfinite(clf.outlier_scores_)], rug=True)
    plot.show()

    #Represent results?
    represent=True
    if(represent==True):
        # # Represent outliers
        # MDS Reduction in 2-dimensions space
        print('Start MDS reduction to represent outliers in 2D space, Please wait...')
        attributes = mds_reduction2d(attributes)
        fig = plot.figure(figsize=plot.figaspect(1/3))
        hdbo_axis = fig.add_subplot('131')
        hdbo_z_axis = fig.add_subplot('132')
        hdbo_zz_axis = fig.add_subplot('133')
        hdbo_axis.scatter(*attributes.T, linewidth=0, c='gray', alpha=0.25)
        hdbo_axis.scatter(*attributes[outliers].T, linewidth=0, c='red', alpha=0.5)
        # Create a Rectangle patch
        rect = patches.Rectangle((-10,-10),20,20,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        hdbo_axis.add_patch(rect)
        hdbo_axis.set_title('HDBSCAN Outliers')
        hdbo_axis.set_xlabel('Min_cluster_size: %s' % str(min_cluster_size) + '\nMin_samples: %s' % str(min_samples))

        # Zoomed
        hdbo_z_axis.set_xlim((-10, 10))
        hdbo_z_axis.set_ylim((-10, 10))
        hdbo_z_axis.scatter(*attributes.T, linewidth=0, c='gray', alpha=0.25)
        hdbo_z_axis.scatter(*attributes[outliers].T, linewidth=0, c='red', alpha=0.5)
        hdbo_z_axis.set_title('HDBSCAN Outliers (zoomed)')
        hdbo_z_axis.set_xlabel('Estimated number of outliers: %d' % len(outliers))
          # Create a Rectangle patch
        rect = patches.Rectangle((-2.5,-2.5),5,5,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        hdbo_z_axis.add_patch(rect)
        # Extra zoomed
        hdbo_zz_axis.set_xlim((-2.5, 2.5))
        hdbo_zz_axis.set_ylim((-2.5, 2.5))
        hdbo_zz_axis.scatter(*attributes.T, linewidth=0, c='gray', alpha=0.25)
        hdbo_zz_axis.scatter(*attributes[outliers].T, linewidth=0, c='red', alpha=0.5)
        hdbo_zz_axis.set_title('HDBSCAN Outliers (Extra zoomed)')
        #hdbo_zz_axis.set_xlabel('Estimated number of outliers: %d' % len(outliers))
        plot.show()

        # Black removed and is used for noise instead.
        hdb_unique_labels = set(hdb_labels)
        hdb_colors = plot.cm.Spectral(np.linspace(0, 1, len(hdb_unique_labels)))
        # Represent HDBSCAN results
        fig = plot.figure(figsize=plot.figaspect(1/3))
        hdb_axis = fig.add_subplot('131')
        hdb_z_axis = fig.add_subplot('132')
        hdb_zz_axis = fig.add_subplot('133')
        for k, col in zip(hdb_unique_labels, hdb_colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
            hdb_axis.scatter(attributes[hdb_labels == k, 0], attributes[hdb_labels == k, 1], marker='o', color=col, alpha=0.3, edgecolors='face')
        hdb_axis.set_title('HDBSCAN\nEstimated number of clusters: %d' % n_clusters_hdb_)
        hdb_axis.set_xlabel('Min_cluster_size: %s' % str(min_cluster_size) + '\nMin_samples: %s' % str(min_samples))
        # Create a Rectangle patch
        rect = patches.Rectangle((-10,-10),20,20,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        hdb_axis.add_patch(rect)
        # Subplot with zoomed HDBSCAN results
        for k, col in zip(hdb_unique_labels, hdb_colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
            hdb_z_axis.scatter(attributes[hdb_labels == k, 0], attributes[hdb_labels == k, 1], marker='o', color=col, alpha=0.3, edgecolors='face')
        hdb_z_axis.set_xlim((-10, 10))
        hdb_z_axis.set_ylim((-10, 10))
        hdb_z_axis.set_title('HDBSCAN (Zoomed)\n')
        hdb_z_axis.set_xlabel('Estimated number of outliers: %d' % len(outliers))
          # Create a Rectangle patch
        rect = patches.Rectangle((-2.5,-2.5),5,5,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        hdb_z_axis.add_patch(rect)
        # Subplot with Extra zoomed HDBSCAN results
        for k, col in zip(hdb_unique_labels, hdb_colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
            hdb_zz_axis.scatter(attributes[hdb_labels == k, 0], attributes[hdb_labels == k, 1], marker='o', color=col, alpha=0.3, edgecolors='face')
        hdb_zz_axis.set_xlim((-2.5, 2.5))
        hdb_zz_axis.set_ylim((-2.5, 2.5))
        hdb_zz_axis.set_title('HDBSCAN (Extra Zoomed)\n')
        #hdb_zz_axis.set_xlabel('Estimated number of outliers: %d' % len(outliers))
        plot.show()

    return clf

def lof_clustering(attributes):
    lof_t1 = time.time()
    # Local Outlier Factor
    # fit the model for outlier detection (default)
    clf = neighbors.LocalOutlierFactor(n_neighbors=20, n_jobs=-1)
    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).
    y_pred = clf.fit_predict(attributes)
    lof_elapsed_time = time.time() - lof_t1
    X_scores = clf.negative_outlier_factor_
    ##########################################
    # OPTIMIZATION ID: OP003
    ##########################################
    lof_score = clf.negative_outlier_factor_
    # Standardize score
    lof_score = 2.*(lof_score - np.min(lof_score))/np.ptp(lof_score)-1
    X_scores = lof_score
    outliers = np.asarray(np.where(lof_score < 0))
    ##########################################
    # END OPTIMIZATION
    ##########################################
    # Next commented lines are replaced with optimization lines
    #threshold = pd.Series(clf.negative_outlier_factor_).quantile(0.05)
    #outliers = np.asarray(np.where(clf.negative_outlier_factor_ < threshold)[0])
    print('\n\n++++ LOF Results ++++')
    print('Elapsed time to outlier detection: %.4f s' % lof_elapsed_time)
    print('Estimated number of outliers: ', outliers.size)

    #Represent results?
    represent=True
    if(represent==True):
        # # Represent outliers
        # MDS Reduction in 2-dimensions space
        print('Start MDS reduction to represent outliers in 2D space, Please wait...')
        attributes = mds_reduction2d(attributes)
        fig = plot.figure(figsize=plot.figaspect(1/3))
        lof_axis = fig.add_subplot('131')
        lof_z_axis = fig.add_subplot('132')
        lof_zz_axis = fig.add_subplot('133')
        # plot circles with radius proportional to the outlier scores
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        # First subplot
        lof_axis.set_title("LOF Outliers")
        lof_axis.set_xlabel('Estimated number of outliers: %d' % outliers.size)
        lof_axis.scatter(attributes[:, 0], attributes[:, 1], color='k', s=3., label='Data points')
        lof_axis.scatter(attributes[:, 0], attributes[:, 1], s=1000 * radius, edgecolors='r',facecolors='none', label='Outlier scores')
        legend = lof_axis.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]
        # Create a Rectangle patch
        rect = patches.Rectangle((-10,-10),20,20,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        lof_axis.add_patch(rect)
        # Second subplot (zoomed)
        lof_z_axis.set_title("LOF Outliers (Zoomed)")
        lof_z_axis.set_xlim((-10, 10))
        lof_z_axis.set_ylim((-10, 10))
        lof_z_axis.scatter(attributes[:, 0], attributes[:, 1], color='k', s=3., label='Data points')
        lof_z_axis.scatter(attributes[:, 0], attributes[:, 1], s=1000 * radius, edgecolors='r',facecolors='none', label='Outlier scores')
        legend = lof_z_axis.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]
        # Create a Rectangle patch
        rect = patches.Rectangle((-5,-5),10,10,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        lof_z_axis.add_patch(rect)
        # Third subplot (Extra zoomed)
        lof_zz_axis.set_title("LOF Outliers (Extra Zoomed)")
        lof_zz_axis.set_xlim((-5, 5))
        lof_zz_axis.set_ylim((-5, 5))
        lof_zz_axis.scatter(attributes[:, 0], attributes[:, 1], color='k', s=3., label='Data points')
        lof_zz_axis.scatter(attributes[:, 0], attributes[:, 1], s=1000 * radius, edgecolors='r',facecolors='none', label='Outlier scores')
        legend = lof_zz_axis.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]

        plot.show()

    return clf

def hdbscan_prediction(classificator, attributes, new_attributes):
    hdbp_t1 = time.time()
    # Predict new data with classification algorithm trained
    test_labels, strengths = hdbscan.approximate_predict(classificator, new_attributes)
    hdbp_elapsed_time = time.time() - hdbp_t1
    # Number of clusters in test_labels and classificator.lables_, ignoring noise if present.
    n_clusters_predicted_ = len(set(test_labels)) - (1 if -1 in test_labels else 0)
    n_clusters = len(set(classificator.labels_)) - (1 if -1 in classificator.labels_ else 0)
    n_noise = list(test_labels).count(-1)
    n_assig_cluster = len(list(test_labels)) - n_noise
    print('\n\n++++ HDBSCAN Prediction Results ++++')
    print('Estimated number of clusters assigned to new data: %d' % n_clusters_predicted_ + '/' + str(n_clusters))
    print('Estimated number of assigned cluster points: %d' % n_assig_cluster)
    print('Estimated number of new noise/outliers points: %d' % n_noise)
    print('Elapsed time to predict ' + str(len(new_attributes)) + ' new data entries: %.4f s' % hdbp_elapsed_time)

    #Represent results?
    represent=True
    if(represent==True):
        # Represent predicted data
        # MDS Reduction in 2-dimensions space
        print('Start MDS reduction to represent outliers in 2D space, Please wait...')
        attributes = mds_reduction2d(attributes)
        new_attributes = mds_reduction2d(new_attributes)
        # Color Palette
        pal = sns.color_palette('deep', n_colors=n_clusters)
        colors = [sns.desaturate(pal[col], sat) for col, sat in zip(classificator.labels_, classificator.probabilities_)]
        test_colors = [pal[col] if col >= 0 else (0.1, 0.1, 0.1) for col in test_labels]
        # Represent prediction
        fig = plot.figure(figsize=plot.figaspect(1/3))
        hdbp_axis = fig.add_subplot('131')
        hdbp_z_axis = fig.add_subplot('132')
        hdbp_zz_axis = fig.add_subplot('133')
        # First subplot
        hdbp_axis.set_title('HDBSCAN Prediction')
        hdbp_axis.scatter(attributes.T[0], attributes.T[1], c=colors, label='Train Data');
        hdbp_axis.scatter(*new_attributes.T, c=test_colors, linewidths=1, edgecolors='k', label='Test data')
        # Create a Rectangle patch
        rect = patches.Rectangle((-10,-10),20,20,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        hdbp_axis.add_patch(rect)
        # Segond subplot (zoomed)
        hdbp_z_axis.set_title('HDBSCAN Prediction (Zoomed)')
        hdbp_z_axis.set_xlim((-10, 10))
        hdbp_z_axis.set_ylim((-10, 10))
        hdbp_z_axis.scatter(attributes.T[0], attributes.T[1], c=colors);
        hdbp_z_axis.scatter(*new_attributes.T, c=test_colors, linewidths=1, edgecolors='k')
        # Create a Rectangle patch
        rect = patches.Rectangle((-2.5,-2.5),5,5,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        hdbp_z_axis.add_patch(rect)
        # Third subplot (Extra zoomed)
        hdbp_zz_axis.set_title('HDBSCAN Pred. (Extra Zoomed)')
        hdbp_zz_axis.set_xlim((-2.5, 2.5))
        hdbp_zz_axis.set_ylim((-2.5, 2.5))
        hdbp_zz_axis.scatter(attributes.T[0], attributes.T[1], c=colors);
        hdbp_zz_axis.scatter(*new_attributes.T, c=test_colors, linewidths=1, edgecolors='k')
        plot.show()

    return test_labels,strengths

def lof_prediction(attributes, new_attributes):
    lofp_t1 = time.time()
    # fit the model for novelty detection
    clf = neighbors.LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    clf.fit(attributes)
    # DO NOT use predict, decision_function and score_samples on X_train as this
    # would give wrong results but only on new unseen data (not used in X_train),
    # e.g. X_test, X_outliers or the meshgrid
    # Predict new data with classification algorithm trained
    attr_pred = clf.predict(new_attributes)
    n_outliers = attr_pred[attr_pred == -1].size
    n_inliers = attr_pred[attr_pred == 1].size
    lofp_elapsed_time = time.time() - lofp_t1

    print('\n\n++++ Local Outlier Detection Prediction Results ++++')
    print('Estimated number of new outlier points: %d' % n_outliers)
    print('Elapsed time to predict ' + str(len(new_attributes)) + ' new data entries: %.4f s' % lofp_elapsed_time)


    #Represent results?
    represent=True
    if(represent==True):
        # Represent outliers
        fig = plot.figure(figsize=plot.figaspect(1/3))
        lof_axis = fig.add_subplot('131')
        lof_z_axis = fig.add_subplot('132')
        lof_zz_axis = fig.add_subplot('133')
        # MDS Reduction in 2-dimensions space
        print('Start MDS reduction to represent outliers in 2D space, Please wait...')
        attributes = mds_reduction2d(attributes)
        new_attributes = mds_reduction2d(new_attributes)
        # Separate outliers and inliers to represent separately
        new_attr_outliers = np.empty((0,2))
        new_attr_inliers = np.empty((0,2))
        for k, col in zip(new_attributes, attr_pred):
            if col == -1:
                new_attr_outliers = np.vstack((new_attr_outliers,k))
            else:
                new_attr_inliers = np.vstack((new_attr_inliers,k))
        # First subplot
        lof_axis.set_title("LOF Outliers")
        lof_axis.set_xlabel('Estimated number of outliers (red): %d' % n_outliers + '\nEstimated number of inliers (black): %d' % n_inliers)
        lof_axis.scatter(attributes[:, 0], attributes[:, 1], color='gray', s=3., label='Data points')
        lof_axis.scatter(new_attr_outliers[:,0], new_attr_outliers[:,1], color='r', s=6, label='New outlier')
        lof_axis.scatter(new_attr_inliers[:,0], new_attr_inliers[:,1], color='k', s=6, label='New inliers')
        legend = lof_axis.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]
        # Create a Rectangle patch
        rect = patches.Rectangle((-10,-10),20,20,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        lof_axis.add_patch(rect)
        # Second subplot (zoomed)
        lof_z_axis.set_title("LOF Outliers (Zoomed)")
        lof_z_axis.set_xlim((-10, 10))
        lof_z_axis.set_ylim((-10, 10))
        lof_z_axis.scatter(attributes[:, 0], attributes[:, 1], color='gray', s=3., label='Data points')
        lof_z_axis.scatter(new_attr_outliers[:,0], new_attr_outliers[:,1], color='r', s=6, label='New outlier')
        lof_z_axis.scatter(new_attr_inliers[:,0], new_attr_inliers[:,1], color='k', s=6, label='New inliers')
        legend = lof_z_axis.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]
        # Create a Rectangle patch
        rect = patches.Rectangle((-2.5,-2.5),5,5,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        lof_z_axis.add_patch(rect)
        # Second subplot (Extra zoomed)
        lof_zz_axis.set_title("LOF Outliers (Extra Zoomed)")
        lof_zz_axis.set_xlim((-2.5, 2.5))
        lof_zz_axis.set_ylim((-2.5, 2.5))
        lof_zz_axis.scatter(attributes[:, 0], attributes[:, 1], color='gray', s=3., label='Data points')
        lof_zz_axis.scatter(new_attr_outliers[:,0], new_attr_outliers[:,1], color='r', s=6, label='New outlier')
        lof_zz_axis.scatter(new_attr_inliers[:,0], new_attr_inliers[:,1], color='k', s=6, label='New inliers')
        legend = lof_zz_axis.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]

        plot.show()

    return attr_pred

##############
# START MAIN #
##############
# Get connection details from file in dictionary format.
connection_file = './mysql_connection'

# Create Engine connection to MySQL database
engine = mysql_connection(connection_file)

# If Engine connection exist, read Mysql table into a pandas DataFrame
if engine:
    data = pd.read_sql("SELECT det.NumTransaccion,IdUsuario,FechaTransaccion,Valor FROM tesis_trace.det det INNER JOIN tesis_trace.reg reg on det.NumTransaccion=reg.NumTransaccion where IdEvento = 'A' and NombreCampo = 'HISTORIA' and (IdTipoEntidad = 'F' OR IdTipoEntidad = 'D') ORDER BY reg.NumTransaccion DESC LIMIT 9000", engine)
    # Separate a sample from data to predict new data
    rows_to_predict = 1000
    prediction_data = data.tail(rows_to_predict)
    data = data.head(len(data.index)-rows_to_predict)

    # Transform DataFrame into a numerical array to works with model
    dataset = preprocessing_dataset(data)

    # Data standardization
    dataset = data_standardization(dataset)

    reduceDimension=False
    if reduceDimension==True:
        # Dimensionality reduction with PCA
        X_pca = pca_decomposition(dataset)

        # Dimensionality reduction with SVD
        X_svd = svd_decomposition(dataset)

        # Represent dimensionality reduction methods
        mds = manifold.MDS(n_components=2, n_init=4, n_jobs=-1)
        X_mds = mds.fit_transform(dataset)
        X_pca_mds = mds.fit_transform(X_pca)
        X_svd_mds = mds.fit_transform(X_svd)
        fig, (ax1,ax2,ax3) = plot.subplots(nrows=1,ncols=3, figsize=(30,6))
        ax1.set_title('Original Data')
        ax1.scatter(X_mds[:, 0], X_mds[:, 1])
        ax2.set_title('PCA decomposition')
        ax2.scatter(X_pca_mds[:, 0], X_pca_mds[:, 1])
        ax3.set_title('SVD decomposition')
        ax3.scatter(X_svd_mds[:, 0], X_svd_mds[:, 1])
        plot.show()

    X=dataset
    ##########################################
    # OUTLIER DETECTION in Historichal data
    ##########################################

    # HDBSCAN Outlier detection in historical data
    hdbs = hdbscan_clustering(X)
    # Transform NaN values to 0 because the order move NaN values to top of list.
    hdbs_scores = np.nan_to_num(hdbs.outlier_scores_)
    # Get top 10 outliers in descendent order.
    idx1 = hdbs_scores.argsort()[::-1][:10]
    print("Top 10 max scores:\n",hdbs_scores[idx1])
    print("Corresponent positions in dataset of top 10 max scores:\n", idx1)
    print("Top 10 Outliers using HDBSCAN method:\n", data.iloc[idx1])
    # Get outliers
    outliers = np.asarray(np.where(hdbs_scores > 0))
    hdbscan_outliers = data.iloc[outliers[0]]
    # Represent distribution of outliers score
    plot.hist(hdbs_scores)
    # Add score to outliers
    hdbscan_outliers['HDBS_score'] = hdbs_scores[outliers[0]].tolist()
    hdbscan_outliers = hdbscan_outliers.sort_values(ascending=False, by='HDBS_score')

    # LOF Outlier detection
    lof = lof_clustering(X)
    idx2 = lof.negative_outlier_factor_.argsort()[:10]
    print("Top 10 max scores (in negative form):\n",lof.negative_outlier_factor_[idx2])
    print("Corresponent positions in dataset of top 10 max scores:\n", idx2)
    print("Top 10 Outliers using LOF method:\n", data.iloc[idx2])
    ##########################################
    # OPTIMIZATION ID: OP003
    ##########################################
    lof_score = lof.negative_outlier_factor_
    # Standardize score
    lof_score = 2.*(lof_score - np.min(lof_score))/np.ptp(lof_score)-1
    # Get outliers
    outliers = np.asarray(np.where(lof_score < 0)[0])
    # Represent distribution of scaled scores and threshold used
    threshold = 0  # Scores over 0 are inlier, below are outlier.
    sns.distplot(lof_score, hist=False)
    plot.axvline(threshold, color='r')
    plot.show()
    ##########################################
    # END OPTIMIZATION
    ##########################################
    # Next commented lines are replaced with optimization lines
    #threshold = pd.Series(lof.negative_outlier_factor_).quantile(0.05)
    #outliers = np.asarray(np.where(lof.negative_outlier_factor_ < threshold)[0])
    outliers_score = lof.negative_outlier_factor_[outliers]
    # Represent distribution of outliers score
    # Next commented lines are replaced with optimization lines
    #sns.distplot(outliers_score, hist=False)
    #plot.axvline(threshold, color='r')
    #plot.show()
    # Add score to outliers
    lof_outliers = data.iloc[outliers]
    lof_outliers['LOF_score'] = outliers_score.tolist()
    lof_outliers = lof_outliers.sort_values(by='LOF_score')

    # Compare results
    n_outliers_hdbscan = np.asarray(np.where(hdbs_scores > 0)).size
    ##########################################
    # OPTIMIZATION ID: OP003
    ##########################################
    outliers = np.asarray(np.where(lof_score < 0)[0])
    n_outliers_lof = outliers.size
    same_outliers = np.intersect1d(hdbs_scores.argsort()[::-1][:n_outliers_hdbscan], lof_score.argsort()[:n_outliers_lof])
    dif_outliers = np.setdiff1d(hdbs_scores.argsort()[::-1][:n_outliers_hdbscan], lof_score.argsort()[:n_outliers_lof])
    ##########################################
    # END OPTIMIZATION
    ##########################################
    # Next commented lines are replaced with optimization lines
    #threshold = pd.Series(lof.negative_outlier_factor_).quantile(0.05)
    #outliers = np.asarray(np.where(lof.negative_outlier_factor_ < threshold)[0])
    # n_outliers_lof = outliers.size
    # same_outliers = np.intersect1d(hdbs_scores.argsort()[::-1][:n_outliers_hdbscan], lof.negative_outlier_factor_.argsort()[:n_outliers_lof])
    # dif_outliers = np.setdiff1d(hdbs_scores.argsort()[::-1][:n_outliers_hdbscan], lof.negative_outlier_factor_.argsort()[:n_outliers_lof])
    print('Same Outliers detected between HDBSCAN and LOF: %d' % same_outliers.shape)
    print('Different Outliers detected between HDBSCAN and LOF: %d' % dif_outliers.shape)

    ##########################################
    # Predict new data
    ##########################################
    # Read file with new data
    #prediction_data.to_json(r'new_data.json')
    new_data = pd.read_json(r'new_data.json')
    # Transform DataFrame into a numerical array to works with model
    new_dataset = preprocessing_dataset(new_data)
    # New data standardization
    new_dataset = data_standardization(new_dataset)

    X_new = new_dataset

    # HDBSCAN predict new data
    test_labels,strengths = hdbscan_prediction(hdbs, X, X_new)
    outliers_hdbs = np.where(test_labels == -1)
    hdbs_Poutliers = new_data.iloc[outliers_hdbs]

    # LOF predict new data
    attr_pred = lof_prediction(X, X_new)
    outliers_lof = np.where(attr_pred == -1)
    lof_Poutliers = new_data.iloc[outliers_lof]

    # Compare prediction results
    same_Poutliers = np.intersect1d(outliers_hdbs, outliers_lof)
    dif_Poutliers = np.setdiff1d(outliers_hdbs, outliers_lof)
    print('++++ Compare prediction results ++++')
    print('Same Outliers detected between HDBSCAN and LOF: %d' % same_Poutliers.shape)
    print('Different Outliers detected between HDBSCAN and LOF: %d' % dif_Poutliers.shape)



##############
#  END MAIN  #
##############
