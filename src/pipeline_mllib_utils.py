# pipeline_mllib_utils.py
import numpy as np
from statistics import mean
import pickle
from ast import literal_eval
from time import gmtime, strftime
from collections import Counter
from statistics import mean
import pickle


from pyspark.mllib.linalg import SparseVector as mllib_SparseVector, DenseVector
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel



k100t50 = [173, 747, 794, 922, 1556, 2018, 2175, 2267, 2332, 2339, 2430, 2775, 2847, 3114, 3336, 3498, 4038, 4109, 4124, 4756, 4803, 4955, 5618, 5928, 6496, 6695, 7184, 7283, 7760, 7777, 7852, 7998, 8039, 8708, 8726, 8763, 8783, 10017, 11828, 11941, 11960, 12026, 12209, 12239, 12706, 13074, 13146, 13173, 13349, 13768, 13787, 14213, 14791, 14977, 15240, 15732, 16123, 16307, 16637, 16673, 17003, 17169, 17442, 17598, 17711, 18281, 19152, 19173, 19304, 19336, 19726, 20185, 20444, 20602, 20888, 21050, 21382, 21603, 21881, 21903, 22081, 22899, 22986, 23195, 23256, 23335, 23625, 23694, 24417, 24761, 24955, 25284, 25632, 25902, 26667, 26858, 27295, 27375, 27510, 27520, 27566, 27826, 27889, 28051, 28151, 28332, 28408, 28587, 28785, 29239, 29579, 29587, 30078, 30413, 30448, 30759, 30937, 31222, 31350, 31650, 31718, 31843, 32176, 32397, 32423, 32437, 32631, 32743, 32885, 32943, 33288, 33299, 33443, 33465, 33987, 33993, 34169, 34225, 34996, 35046, 35450, 35771, 36043, 36330, 36541, 36989, 37029, 37058, 37577, 37597, 37632, 37838, 38028, 38476, 38928, 38943, 38988, 39202, 39569, 39594, 40183, 40711, 40799, 41055, 41392, 42072, 42307, 42468, 42578, 42582, 42970, 43165, 44866, 45268, 45539, 45571, 45802, 46064, 46115, 46182, 46539, 46784, 47248, 47877, 47977, 48400, 48465, 49031, 49425, 49791, 49813, 50097, 50418, 50445, 50558, 50606, 50609, 50711, 50750, 50840, 51071, 51326, 51376, 51454, 51817, 52120, 52430, 53353, 54219, 54222, 54391, 55076, 55380, 55625, 55945, 56709, 57055, 58747, 58791, 59146, 59220, 59331, 59715, 60072, 60381, 60606, 61172, 62098, 62220, 62787, 63176, 63515, 63679, 63796, 64622, 64937, 64967, 65303, 66069, 66539, 67298, 67670, 67753, 67816, 68394, 68422, 69575, 69652, 69937, 70150, 70331, 70811, 70980, 71119, 71476, 71753, 72082, 72460, 72597, 72968, 73442, 73863, 74258, 74267, 74344, 74870, 74871, 75166, 75183, 75610, 75659, 75725, 75854, 76213, 76310, 77193, 77442, 77462, 77511, 77531, 78354, 79106, 79225, 79590, 79709, 79750, 79784, 79964, 80799, 81483, 81531, 81562, 81569, 81761, 82038, 82330, 82399, 82608, 82712, 82981, 83155, 83505, 84301, 84332, 84461, 84474, 84629, 85508, 86191, 86262, 86312, 86528, 86703, 87240, 87266, 87354, 87356, 87371, 88217, 88255, 88923, 89855, 90098, 90099, 90561, 91511, 91754, 92138, 92381, 93070, 93361, 93366, 93570, 93571, 93656, 94049, 94491, 94587, 94697, 94791, 94825, 94923, 95589, 96258, 96809, 97066, 97350, 97585, 98073, 98876, 98953, 99175, 99508, 99818, 99888, 99965, 100095, 100737, 100808, 101051, 101126, 101188, 101495, 102042, 102177, 102415, 103889, 104339, 104343, 104383, 104679, 104990, 105288, 105478, 105758, 105857, 105949, 106434, 106537, 108082, 108277, 108573, 108611, 109210, 109284, 109476, 109501, 109562, 109918, 111149, 111331, 111962, 112008, 112233, 113094, 113109, 113141, 113142, 113289, 113355, 113461, 113549, 113705, 114415, 114523, 114624, 114783, 114961, 115111, 115362, 115640, 115871, 115952, 116394, 117478, 117872, 118422, 118544, 118637, 118724, 119012, 119035, 119217, 119243, 119454, 119947, 120128, 120271, 120445, 120612, 121057, 121172, 121415, 121967, 122169, 122436, 122438, 123107, 123988, 124495, 124533, 124718, 125299, 125976, 126597, 126692, 127037, 127041, 128830, 128992, 129496, 129842, 130048, 130674, 130686, 131156, 131295, 132405, 132757, 132769, 132810, 133411]


def timestamp():
    return strftime('%Y-%m-%d %H:%M:%S', gmtime())

def get_best_classifier(sc, data_dir):
    # model_randfor_class_t04_d04_b024_s7593
    # model_randfor_class_t04_d04_b024_s7593_params.pckl
    from os import listdir
    from os.path import isfile, join
    models = [(f,int(f[-4:])) for f in listdir(data_dir) if f[:6] == 'model_' and len(f) == 39]
    best_model = max(models,key=lambda item:item[1])
    print('best classifier found is |{0}| score = {1} '.format(*best_model))
    params = load_pickle_file(data_dir + best_model[0] + '_params.pckl')
    print(params)
    #return (RandomForestClassificationModel.load(data_dir + best_model[0]), params)
    return (RandomForestModel.load(sc=sc, path=data_dir+best_model[0]), params)

def get_best_classifier_params(data_dir):
    # model_randfor_class_t04_d04_b024_s7593
    # model_randfor_class_t04_d04_b024_s7593_params.pckl
    from os import listdir
    from os.path import isfile, join
    models = [(f,int(f[35:39])) for f in listdir(data_dir) if f[:6] == 'model_' and len(f) == 51]
    best_model = max(models,key=lambda item:item[1])
    print('best classifier found is |{0}| score = {1} '.format(*best_model))
    params = load_pickle_file(data_dir + best_model[0])
    print(params)
    return (params, best_model[0]) # return params and model name


# MISC EDA
def count_houses_with_particular_feat(f_rdd, feat_list):
    feat_list = set(feat_list)
    return f_rdd.filter(lambda x: any([f[0] in feat_list for f in x[2]]) ).count()

### MANUAL UTILS SECTION ###
def load_pickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
def write_pickle_file(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
def calc_feature_sd(rdd):
    sd_rdd = rdd.map(lambda x: [(t[0], [t[1]]) for t in x[2]]) \
                       .flatMap(lambda x: x) \
                       .reduceByKey(lambda a, b: a + b)
    sd_rdd = sd_rdd.map(lambda x: (x[0], np.std(x[1], ddof=1) ))
    sd = dict(sd_rdd.collect())
    return sd

###    CODE ###
def count_features(rdd):
    ''' count features across all HHs '''
    # e.g. {133524, 27} --> feat 133524 on 27 HHs
    # takes ~45 seconds on 64 cores
    feat_count_rdd = rdd.map(lambda x: [(t[0],1) for t in x[2]]) \
                        .flatMap(lambda x: x) \
                        .reduceByKey(lambda a, b: a+b)
    feat_counts = dict(feat_count_rdd.collect())
    return feat_counts


# def create_features_rdd(base_rdd, feat_count, sparsity_cutoff, bool_cutoff, insert_means = False):
#     ''' removes sparse features, boolifies others, inserts mean values '''
#     insert_means = False  # NOT IMPLEMENTED
#     bool_cutoff = max(bool_cutoff, sparsity_cutoff)  # bool cutoff can not be less than sparsity cutoff
#     feat_to_keep = get_feat_above_cutoff(feat_count, sparsity_cutoff)
#     feat_to_bool = set(feat_to_keep) - set(get_feat_above_cutoff(feat_count, bool_cutoff))
# 
#     print('  # of features to keep = {0}'.format(len(feat_to_keep)))
#     print('  # of features to bool = {0}'.format(len(feat_to_bool)))
# 
#     # creation of features_rdd
#     features_rdd, map_new_old_featnum = filter_features(base_rdd, feat_to_keep)
#     features_rdd = bool_features(features_rdd, feat_to_bool)
# 
#     if insert_means:
#         feat_to_mean = set(feat_to_keep) - set(feat_to_bool)
#         features_rdd = mean_features(features_rdd, feat_means, feat_to_mean)
# 
#     # return (features_rdd, map_new_old_featnum, feat_to_keep, feat_to_bool)
#     return (features_rdd, map_new_old_featnum, feat_to_keep)


def get_feat_above_cutoff(record_count, feat_counts_dict, cutoff=.20):
    ''' get list of feat# above a certain cutoff
        used for feat_to_keep and feat_to_bool
    :returns: list of feat#
    '''
    cutoff = record_count * cutoff
    feat_list = [k for k in feat_counts_dict if feat_counts_dict[k] > cutoff]
    return feat_list

def filter_features(feat_rdd, feat_to_keep):
    ''' filters a feat_rdd keeping only the feature on the feat_list
    renumbers/indexes features in accordance with new length
    :param feat_rdd:  a features_rdd formatted rdd
    :param feat_list:  list of ints, features to keep
    :returns:  filtered_rdd,  map_new_old_featnum, map_old_new_featnum
    takes < 1 min on 64 cores
    '''
    feat_to_keep_set = set(feat_to_keep)
    try:
        feat_to_keep.sort()
    except:
        print('warning: filter_features got a non-list again')
    map_new_old_featnum = {k:v for k,v in zip(range(len(feat_to_keep)),feat_to_keep)}
    map_old_new_featnum = {k:v for k,v in zip(feat_to_keep,range(len(feat_to_keep)))}
    filt_rdd = feat_rdd.map(lambda x: (x[0], [f for f in x[2] if f[0] in feat_to_keep_set]) ) \
                       .filter(lambda x: len(x[1]) > 0) \
                       .map(lambda x: (x[0], len(feat_to_keep), x[1]) ) \
                       .map(lambda x: (x[0], x[1], [(map_old_new_featnum[f[0]],f[1]) for f in x[2]]) )
    # don't return map_old_new since we don't seem to need it
    return (filt_rdd, map_new_old_featnum)

def filter_for_reqd_feat(rdd, required_feats):
    ''' returns only housholes that have at least 1 of the required feats 
        generally used for the pca identified features 
        this does NOT renumber features
    '''
    return rdd.filter(lambda x: any([f[0] in required_feats for f in x[2]]) )

def bool_features(feat_rdd, feat_to_bool):
    ''' filteres a feat_rdd keeping converting features to bool
    :returns:  filtered_rdd
    '''
    feat_to_bool_set = set(feat_to_bool)
    def process_row_feats(orig_feats):
        orig_feats_set = set([f[0] for f in orig_feats])
        # keep all the other feats unchanged
        new_feats = [f for f in orig_feats if f[0] not in feat_to_bool_set]
        # set present bool feats to 1
        new_feats = new_feats + [(f[0],1) for f in orig_feats if f[0] in feat_to_bool_set]
        # set absent bool feats to 0
        new_feats = new_feats + [(f,0) for f in feat_to_bool_set if f not in orig_feats_set]
        new_feats.sort()
        return new_feats
                
    filt_rdd = feat_rdd.map(lambda x: (x[0], x[1], process_row_feats(x[2])) )
    return filt_rdd

# for PCA results
def top_predictors( k, uv, topx, translator=None):
    # k: the k we can pca with / number of s groups
    # uv: ussually u or v.T, an nXm np.ndarray
    # topx is the number of results to return for each k group
    assert k == uv.shape[0] 
    answer = {}
    for i in range(k):
        col = uv[i,:]
        idx_topx = col.argsort()[-topx:][::-1]
        if translator:
            answer[i] = [translator[x] for x in idx_topx]
        else:
            answer[i] = [x for x in idx_topx]
    return answer

def remap_map(old, new):
    # returns a new dict mapped to values in old dict
    for k in new_dict:
        new_dict[k] = old_dict[new_dict[k]]
        

### end of is_spender_pipe 



# def feat_rdd_to_labeledpoints_rdd(f_rdd, make_label_binary=False):
def created_labeledpoint_rdd(f_rdd, make_label_binary=False):
    ''' LabeledPoint(label, features)
          label – Label for this data point.
          features – Vector of features for this point (NumPy array, list, pyspark.mllib.linalg.SparseVector).
         f_rdd should be: label/adspend, max#feat, feat_list 
         LabeledPoint is: label, SparseVector 
         returns rdd of labeled points
         make_label_binary switches the label to true/false
    '''
    if make_label_binary:
        return f_rdd.map(lambda row: LabeledPoint(row[0] > 0, mllib_SparseVector( row[1], dict(row[2]))) )
    else:
        return f_rdd.map(lambda row: LabeledPoint(row[0], mllib_SparseVector( row[1], dict(row[2]))) )

def mllib_rf_class(lp_train_rdd, lp_test_rdd, trees, depth, bins):
    ''' RandomForest Classification
    takes in train/test LabeledPoint rdds
    '''
    model = RandomForest.trainClassifier(lp_train_rdd, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=trees, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=depth, maxBins=bins)

    # Evaluate model on test instances and compute test error
    predictions = model.predict(lp_test_rdd.map(lambda x: x.features))
 
    labelsAndPredictions = lp_test_rdd.map(lambda lp: lp.label).zip(predictions)
    test_error = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(lp_test_rdd.count())
    # print('Learned classification forest model:')
    # print(model.toDebugString())
    return test_error

def mllib_rf_regress(lp_train_rdd, lp_test_rdd, trees, depth, bins):
    ''' RandomForest Regression
    takes in train/test LabeledPoint rdds
    '''
    model = RandomForest.trainRegressor(lp_train_rdd, categoricalFeaturesInfo={},
                                    numTrees=trees, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=depth, maxBins=bins)

    # Evaluate model on test instances and compute test error
    predictions = model.predict(lp_test_rdd.map(lambda x: x.features))
    labelsAndPredictions = lp_test_rdd.map(lambda lp: lp.label).zip(predictions)
    test_error = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() / float(lp_test_rdd.count())
    return test_error
    #print('Test Mean Squared Error = ' + str(testMSE))
    #print(' results trees:{0} depth:{1}, bins:{2}, mse:{3}'.format(trees, depth, bins, testMSE))
    #     print('Learned regression forest model:')
    #     print(model.toDebugString())


def calc_feature_means(rdd):
    ''' calculates the mean for all features in rdd
    VERY slow
    :param rdd: features_rdd
    :return dict: dict of means, e.g. {133524, .00523} --> feat 133524 mean is .00523
    '''
    feat_mean_rdd = rdd.map(lambda x: [(t[0], [t[1]]) for t in x[2]]) \
                       .flatMap(lambda x: x) \
                       .reduceByKey(lambda a, b: a + b) \
                       .map(lambda x: (x[0], mean(x[1]) ))
    feat_means = dict(feat_mean_rdd.collect())
    return feat_means


def mean_features(feat_rdd, feat_means, feat_to_mean):
    def add_means(row):
        have = set([f[0] for f in row[2]])
        need = feat_to_mean - have
        return (row[0], row[1], row[2] + [(x, feat_means[x]) for x in need])
    meaned_rdd = feat_rdd.map(add_means)
    return meaned_rdd

def insert_missing_as_avg(row, avg_values):
    '''
        goes through f_rdd inserting missing values with the avg
    '''
    def add_missing_features(row):
        feat_need = set(range(0,row[1])) - set(f[0] for f in row[2])
        features = sorted(base[2] + [(f, avg_values[f]) for f in feat_need])
        return (row[0], row[1], features)
    return f_rdd.map(add_missing_features)

def center_values(rdd, feat_means, translator):
    ''' returns an rdd with the feature values centerered (values - mean)
        translator maps new->old feat # in rdd(new) to feat_means(old) '''
    return rdd.map(lambda x: (x[0], x[1], \
                    [( f[0], f[1]-feat_means[translator[f[0]]]) for f in x[2]]) ) 






####################
### CRAP SECTION ###
####################

def plot_feature_stats(hh_feat_count, feature_counts):
    ''' feature count by hh '''
    for b in [100,500,1000]:
        fig, ax = plt.subplots( figsize=(14,8) )
        ax.hist(hh_feat_count, b, facecolor='blue', alpha=0.75)
        ax.set_xlabel('features', fontsize=18)
        ax.set_ylabel('households', fontsize=18)
        ax.set_title('feature count by household', fontsize=24)
        # ax.set_xscale('log')
        #ax.set_yscale('log')
        #ax.axvline(x=max(hh_feat_count), linestyle='dashed')
        #ax.set_xticks( xticks + [max(hh_feat_count)] )
        ax.grid(True)
        fig.savefig('images/feature_count_by_hh_' + str(b) + '.png')
    # scatter plot of same
    fig, ax = plt.subplots( figsize=(14,8) )
    fc_count = Counter(hh_feat_count )
    ax.scatter([x for x in fc_count.values()], [x for x in fc_count.keys()], s=1 )
    ax.set_xlabel('features', fontsize=18)
    ax.set_ylabel('households', fontsize=18)
    ax.set_title('feature count by household', fontsize=24)
    ax.grid(True)
    fig.savefig('images/feature_count_by_hh_scat' + str(b) + '.png')
    ''' feature prevalence - distribution of feature counts '''
    # # feature_counts {(133524, 27)... means feat 133524 on 27 HHs
    # fc_count = Counter( [x for x in feature_counts.values()] )
    # # counts for particular features count, how ofter that feature count occur
    # fig, ax = plt.subplots( figsize=(14,8) )
    # 
    # ax.plot([x for x in fc_count.most_common.values()],
    #            [x for x in fc_count.most_common.keys()] )
    # ax.set_ylabel('# of features', fontsize=18)
    # ax.set_xlabel('feature frequency: households with a feature', fontsize=18)
    # ax.set_title('feature prevalence', fontsize=24)
    # ax.grid(True)
    # fig.savefig('images/feature_prevalence.png')
    # 


# TO DELETE
# def remove_sparse_feat(base_rdd, feat_counts_dict, sparsity_cutoff=.20):
#     record_count = base_rdd.count()
#     def filter_features(row):
#         ''' returns a tuple with adspend, feat_list. hh_id is removed '''
#         return (row[1], [f for f in row[2] if f[0] in feat_to_keep_set])
#     cutoff = record_count * sparsity_cutoff
#     feat_to_keep = [k for k, v in feat_counts_dict.items() if v > cutoff]
#     feat_to_keep_set = set(feat_to_keep)
#     feat_to_keep.sort()
#     map_new_old_featnum = {k:v for k,v in zip(range(len(feat_to_keep)),feat_to_keep)}
#     map_old_new_featnum = {k:v for k,v in zip(feat_to_keep,range(len(feat_to_keep)))}
#     print('# of features to keep = {0}'.format(len(feat_to_keep)))
#     # final rdd returns has: adspend, max#feat, feat_list:[(k,v)..]
#     return (base_rdd.map(filter_features) \
#                    .filter(lambda x: len(x[2]) > 0) \
#                    .map(lambda x: (x[0], len(feat_to_keep), x[2]) ) \
#                    .map(lambda x: (x[0], x[1], [(map_old_new_featnum[f[0]],f[1]) for f in x[2]]) ), \
#             map_new_old_featnum, map_old_new_featnum)


# OVERCOMPLICATED & BAD PERFORMANCE
# def create_feature_stats(rdd):
#     ''' returns dataframe with: feature, mean, var, min, max, unqiue, count '''
#     # only take second item from rdd (the feature list), flat map them out
#     stats_rdd = rdd.map(lambda x: x[1]) \
#                    .flatMap(lambda x: x) \
#                    .cache()
#                    # make a copy of stats_rdd to count # of features
#     feat_count_rdd = stats_rdd.map(lambda x: (x[0],1) ) \
#                               .reduceByKey(lambda a, b: a + b)
#     # continue with stats_rdd: split k-v pair, add the k-[v], perform calcs
#     stats_rdd = stats_rdd.map(lambda x: (x[0], [x[1]]) ) \
#                          .reduceByKey(lambda a, b: a + b) \
#                          .map(lambda x: (x[0], float(np.mean(x[1])), \
#                                 float(np.var(x[1])), float(min(x[1])), \
#                                 float(max(x[1])), len(set(x[1]))) )
#     feat_count_schema = StructType([
#         StructField("feature", IntegerType(), True),
#         StructField("num", IntegerType(), True)
#     ])
#     stats_schema = StructType([
#         StructField("feature", IntegerType(), True),
#         StructField("mean", FloatType(), True),
#         StructField("var", FloatType(), True),
#         StructField("min", FloatType(), True),
#         StructField("max", FloatType(), True),
#         StructField("unique", IntegerType(), True)
#     ])    
#     feat_count_df = feat_count_rdd.toDF(feat_count_schema)
#     stats_df = stats_rdd.toDF(stats_schema)
#     return stats_df.join(feat_count_df, 'feature')  # 'feature' solves dup issue


