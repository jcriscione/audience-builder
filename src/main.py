import os
import os.path
import sys

import pyspark as ps
from pyspark.sql import HiveContext
from pyspark.mllib.linalg import SparseVector as mllib_SparseVector, DenseVector
from pyspark.mllib.linalg.distributed import RowMatrix

from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.linalg import VectorUDT

# my code / libs:
from src.BaseETL import BaseETL
from src.pipeline_mllib_utils import *
#from src.pipeline_ml_utils import *

''' global vars: '''
data_dir = 'data/'
feat_means_file = 'feat_means.pckl'
feat_stddevs_file = 'feat_stddevs.pckl'
feat_to_keep_file = 'feat_to_keep.pckl'
pca_top_feat_file = 'pca_top_feat.pckl'
access_key = os.environ['AWS_ACCESS_KEY_ID']
secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
s3_bucket = os.environ['AWS_BUCKET']


def main():
    sys.stdout = open("out_main", "w")  # direct output to file
    print('main starting: {0}'.format(timestamp()), flush=True)
    ''' create the base_rdd '''
    spark = ps.sql.SparkSession.builder \
                  .enableHiveSupport() \
                  .appName('pyspark') \
                  .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    sc = spark.sparkContext
    sqlContext = HiveContext(sc)
    sqlContext.sql("use default")

    print('creating base_rdd, t = {0}'.format(timestamp()), flush=True)
    ''' BaseETL.create_base_rdd()
        :param source:  'file', 's3'
        :param type:    'mini', 'train', 'valid'  (mini is local only)
        :param n:   count  of files to load
        :returns:  base_rdd '''
    base_rdd = BaseETL.create_base_rdd(sc=sc, source='s3', type='train', n=349) \
                      .map(lambda base: (base[1], base[2][0], base[2][1])) \
                      .cache()
    print('finished, t = {0}'.format(timestamp()), flush=True)

    ''' pick pipeline to run '''
    # pipeline_spender(sc, base_rdd)   # grid search randfor classifier
    # pipeline_spend_amt(sc, base_rdd)   # grid search randfor regressor

    print('end of main: {0}'.format(timestamp()))
    sc.stop()    


def pipeline_spender(sc, base_rdd):
    ''' pipeline0 setup '''
    print('  beginning pipeline rf_class, t = {0}'.format(timestamp()), flush=True)
    total_records = base_rdd.count()       # need this to force DAG execution
    print('total_records at beginning of pipeline = {0}'.format(total_records))
 
    feat_count = count_features(base_rdd)
    feat_means = load_pickle_file(data_dir + feat_means_file)
    feat_stddevs = load_pickle_file(data_dir + feat_stddevs_file)
    print('    feat_count len = {0}'.format(len(feat_count)))
    print('    feat_means len = {0}'.format(len(feat_means)))
    print('  feat_stddevs len = {0}'.format(len(feat_stddevs)))

    ''' basic feature reduce '''
    for sparsity_cutoff in [.25, .20, .30, .15]:
        print('beginning features_rdd, t = {0}'.format(timestamp()), flush=True)
        # initial feature reduction: sparse cutoff, just so we can run PCA
        record_count = base_rdd.count()
        feat_to_keep = get_feat_above_cutoff(record_count, feat_count, sparsity_cutoff)
        print('  spar_cutoff = {0} | # of features to keep = {1}'.format(sparsity_cutoff, len(feat_to_keep)))
        features_rdd, map_new_old_featnum = filter_features(base_rdd, feat_to_keep) # filter returning only those allowed

        # DISABLE BOOL CUTOFF AND MEAN INSERTION FOR NOW...
        # bool_cutoff = .20 
        # insert_means = True
        # bool_cutoff = max(bool_cutoff, sparsity_cutoff)  # bool cutoff can not be less than sparsity cutoff
        # feat_to_bool = set(feat_to_keep) - set(get_feat_above_cutoff(feat_count, bool_cutoff))
        # print('  # of features to bool = {0}'.format(len(feat_to_bool)))
        # bool_cutoff = max(bool_cutoff, sparsity_cutoff)  # bool cutoff can not be less than sparsity cutoff
        # features_rdd = bool_features(features_rdd, feat_to_bool)
        # insert_means = False  # NOT IMPLEMENTED
        # if insert_means:
        #     feat_to_mean = set(feat_to_keep) - set(feat_to_bool)
        #     features_rdd = mean_features(features_rdd, feat_means, feat_to_mean)

        #adspend = centered_rdd.map(lambda x: x[0]).collect()

        ''' PCA feature reduce '''
        centered_rdd = center_values(features_rdd, feat_means, map_new_old_featnum )
        features_mat = RowMatrix(centered_rdd.map(lambda x: mllib_SparseVector(x[1], x[2])).cache())

        pca_top_feat = {}
        for k in [10, 25, 50]:
            print('running pca with k = {0}, t = {1}'.format(k, timestamp()), flush=True)
            prin_comp = features_mat.computePrincipalComponents(k)
            for k_n in [25, 50, 100]:
                top_pred = top_predictors( k, prin_comp.toArray().T, k_n ) # new nums, list of lists
                pca_feat_newnum = []
                k_feats_set = set()
                for i in range(k):
                    k_feats_set  = k_feats_set | set([map_new_old_featnum[x] for x in top_pred[i]])
                    #pca_feat_newnum = pca_feat_newnum + [x for x in top_pred[i]]
                pca_top_feat[(k,k_n)] = k_feats_set

        for kkn in pca_top_feat: 
            k = kkn[0]
            k_n = kkn[1]
            print('  creating lp rdd with k={0} and k_n={1}'.format(k,k_n))
            feat_to_keep = pca_top_feat[kkn]
            features_rdd, map_new_old_featnum = filter_features(base_rdd, feat_to_keep) # filter returning only those allowed

            ''' create LabeledPoint rdd for classifier '''
            # random sample to balance pos/neg classes
            lp_rdd = created_labeledpoint_rdd(features_rdd, make_label_binary=True)
            pos = lp_rdd.filter(lambda x: x.label > 0)
            neg = lp_rdd.filter(lambda x: x.label == 0)
            sample_size_goal = 10000
            p = pos.count()
            n = neg.count()
            pos_samp = pos.sample(withReplacement=True, fraction=sample_size_goal/p, seed=42)
            neg_samp = neg.sample(withReplacement=True, fraction=sample_size_goal/n, seed=42)
            # combine the the pos & neg samples, then random split them into train & test
            lp_train_rdd, lp_test_rdd = sc.union([pos_samp, neg_samp]).randomSplit(weights=[0.7, 0.3], seed=42)

            ''' grid search RandomForest classifier '''
            best_score = 0
            for trees in [ 25 ]:
                for depth in [5,10,20,30]:
                    for bins in [4,8,16,32]:
                        print('starting RF classifier, t = {0}'.format(timestamp()))
                        model = RandomForest.trainClassifier(lp_train_rdd, 
                                                numClasses=2, categoricalFeaturesInfo={},
                                                numTrees=trees, featureSubsetStrategy="auto",
                                                impurity='gini', maxDepth=depth, maxBins=bins )
                        predictions = model.predict(lp_test_rdd.map(lambda x: x.features))
                        labelsAndPredictions = lp_test_rdd.map(lambda lp: lp.label).zip(predictions)
                        test_error = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(lp_test_rdd.count())
                        score = 1 - test_error
                        #print('rf_class: trees={0}, depth={1}, bins={2}, score={3}'.format(trees, depth, bins, score))
                        print('rf_class: sc={0}, k={1}, k_n={2}, feat_cnt={3}, trees={4}, depth={5}, bins={6}, score={7}' \
                                .format(sparsity_cutoff, k, k_n, len(feat_to_keep), trees, depth, bins, score), flush=True)
                        try:
                            model_name = 'randfor_class' \
                                                + '_sc' + str(sparsity_cutoff*100).replace('.0','') \
                                                + '_k' + str(k).zfill(3) \
                                                + '_' + str(k_n).zfill(3) \
                                                + '_t' + str(trees).zfill(3) \
                                                + '_d' + str(depth).zfill(2) \
                                                + '_b' + str(bins).zfill(3) \
                                                + '_s' + str(score)[2:6]
                            print('  saving model {0}'.format(model_name))
                            model.save(sc = sc, path = data_dir + model_name)
                            #save parameters in dict for pickling
                            params = {}
                            params['sparsity_cutoff'] = sparsity_cutoff
                            params['k'] = k
                            params['k_n'] = k_n
                            params['feat_to_keep'] = feat_to_keep
                            params['trees'] = trees
                            params['depth'] = depth
                            params['bins'] = bins
                            params['score'] = score
                            write_pickle_file(data_dir + model_name + '_params.pckl', params)
                            # best_score = max(best_score, score)
                        except:
                            print('exception traying to save model {0}'.format(model_name))
    # end pipeline0

def pipeline_spend_amt(sc, base_rdd):
    ''' pipeline1 setup '''
    print('  beginning pipeline #1, t = {0}'.format(timestamp()), flush=True)
    total_records = base_rdd.count()       # need this to force a dag execution
    print('total_records at beginning of pipeline = {0}'.format(total_records))
    feat_count = count_features(base_rdd)
    feat_means = load_pickle_file(data_dir + feat_means_file)

    # rf_class_model_params, class_model_name = get_best_classifier_params(data_dir)
    # for k in rf_class_model_params:
    #     print('  {0} = {1}'.format(k, rf_class_model_params[k]))
    # feat_to_keep = rf_class_model_params['feat_to_keep']

    ''' basic feature reduce '''
    for sparsity_cutoff in [.25, .20, .30, .15]:
        print('beginning features_rdd, t = {0}'.format(timestamp()), flush=True)
        # initial feature reduction: sparse cutoff, just so we can run PCA
        record_count = base_rdd.count()
        feat_to_keep = get_feat_above_cutoff(record_count, feat_count, sparsity_cutoff)
        print('  spar_cutoff = {0} | # of features to keep = {1}'.format(sparsity_cutoff, len(feat_to_keep)))
        features_rdd, map_new_old_featnum = filter_features(base_rdd, feat_to_keep) # filter returning only those allowed

        ''' get just spenders '''
        spenders_rdd = features_rdd.filter(lambda x: x[0] > 0) \
                               .cache()
        ''' PCA to ID best features '''
        centered_rdd = center_values(spenders_rdd, feat_means, map_new_old_featnum )
        features_mat = RowMatrix(centered_rdd.map(lambda x: mllib_SparseVector(x[1], x[2])).cache())
        pca_top_feat = {}
        for k in [25, 10, 50]:
            print('running pca with k = {0}, t = {1}'.format(k, timestamp()), flush=True)
            prin_comp = features_mat.computePrincipalComponents(k)
            for k_n in [25, 50, 100]:
                top_pred = top_predictors( k, prin_comp.toArray().T, k_n ) # new nums, list of lists
                pca_feat_newnum = []
                k_feats_set = set()
                for i in range(k):
                    k_feats_set  = k_feats_set | set([map_new_old_featnum[x] for x in top_pred[i]])
                    #pca_feat_newnum = pca_feat_newnum + [x for x in top_pred[i]]
                pca_top_feat[(k,k_n)] = k_feats_set

        for kkn in pca_top_feat: 
            k = kkn[0]
            k_n = kkn[1]
            print('  creating lp rdd with k={0} and k_n={1}'.format(k,k_n))
            
            base_spendonly_rdd = base_rdd.filter(lambda x: x[0] > 0) \
                                            .cache()
            feat_to_keep = pca_top_feat[kkn]
            spenders_pca_rdd, map_new_old_featnum = filter_features(base_spendonly_rdd, feat_to_keep) # filter returning only those allowed


            ''' create LabeledPoint rdd for classifier '''
            # random sample to balance pos/neg classes
            lp_rdd = created_labeledpoint_rdd(spenders_pca_rdd, make_label_binary=False)
            lp_samp_rdd = lp_rdd.sample(withReplacement=True, fraction=.10, seed=42)
            # combine the the pos & neg samples, then random split them into train & test
            lp_train_rdd, lp_test_rdd = lp_samp_rdd.randomSplit(weights=[0.7, 0.3], seed=42)

            for trees in [ 25 ]:
                for depth in [4,6,10,15,25]:
                    for bins in [4,8,16,32,64]:
                        print('running RF regressor. t={0}'.format(timestamp()))
                        model = RandomForest.trainRegressor(lp_train_rdd, categoricalFeaturesInfo={},
                                                        numTrees=trees, featureSubsetStrategy="auto",
                                                        impurity='variance', maxDepth=depth, maxBins=bins)
                        # Evaluate model on test instances and compute test error
                        predictions = model.predict(lp_test_rdd.map(lambda x: x.features))
                        labelsAndPredictions = lp_test_rdd.map(lambda lp: lp.label).zip(predictions)
                        error = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() / float(lp_test_rdd.count())
                        print('rf_regress: sc={0}, k={1}, k_n={2}, feat_cnt={3}, trees={4}, depth={5}, bins={6}, error={7}' \
                                .format(sparsity_cutoff, k, k_n, len(feat_to_keep), trees, depth, bins, error), flush=True)
                        #print('   lowest mse to beat = {0}'.format(lowest_error))
                        model_name = 'randfor_regress' \
                                            + '_sc' + str(sparsity_cutoff*100).replace('.0','') \
                                            + '_k' + str(k).zfill(3) \
                                            + '_' + str(k_n).zfill(3) \
                                            + '_t' + str(trees).zfill(3) \
                                            + '_d' + str(depth).zfill(2) \
                                            + '_b' + str(bins).zfill(3) \
                                            + '_s' + str(round(error)).zfill(5)
                        print('  saving model {0}'.format(model_name))
                        model.save(sc = sc, path = data_dir + model_name)
                        # save parameters in dict for pickling
                        params = {}
                        params['sparsity_cutoff'] = sparsity_cutoff
                        params['k'] = k
                        params['k_n'] = k_n
                        params['feat_to_keep'] = feat_to_keep
                        params['trees'] = trees
                        params['depth'] = depth
                        params['bins'] = bins
                        params['error'] = error
                        write_pickle_file(data_dir + model_name + '_params.pckl', params)
                        #lowest_error = min(lowest_error, error)
    # end pipeline1


if __name__ == '__main__':
    main()
