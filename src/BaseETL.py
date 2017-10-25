import os
import boto3
from time import gmtime, strftime
import bz2
from ast import literal_eval

try:
    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    s3_bucket = os.environ['AWS_BUCKET']
except:
    print('warning: could not load s3 parameters from os.environ')

class BaseETL(object):
    '''
        reads the data files and creates the base_rdd
    '''

    @staticmethod
    def _read_file_generator_fs(filename):
        ''' reads a data file line-by-line and yields record as python object below
            this method should only be called within rdd.flatMap
            each row is stored (and returned as) a tuple with 3 values:
                t[0] int, household id
                t[1] float, money spent by household in response to advertising
                t[2] tuple, 2 values:
                  t[2][0] int, feature count (133531)
                  t[2][1] list, with features as key-value pairs
            the top-level 3 value tuple is referred to as the "base" object hereafter
            example:
            (40658878, 0.0, (133531, [(51, 0.00440528634361), (158, 4.2850496745097694e-07), ...
        '''
        print('processing file: {0}'.format(filename))
        with bz2.open(filename, 'r') as f:
            for row in f:
                yield literal_eval(row.decode('utf-8')) # the "base" obj

    @staticmethod
    def _read_file_generator_s3(s3filekey):
        ''' similar to _read_file_generator_fs but for AWS S3 
            returns same data structure         '''
        print('processing key: {0}'.format(s3filekey))
        s3client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        s3obj = s3client.get_object(Bucket=s3_bucket, Key=s3filekey)
        byteobj = bz2.decompress(s3obj['Body'].read())
        for row in byteobj.decode('utf-8').strip().split('\n'):
            yield literal_eval(row) # the "base" obj

    @staticmethod
    def create_itemlist_and_generator(source, type, n):
        ''' generates input parameters for all pipelines
            :returns: tuple:
              t[0] a list of local filenames or S3 filekeys to process
              t[1] name of the read_file_generator to use
        '''
        path = 'oa_data/'
        if source == 'file':
            generator_to_use = BaseETL._read_file_generator_fs
            if type == 'mini':
                itemlist = [path+'mini'+str(x).zfill(2)+'.bz2' for x in range(0, n)]
            elif type == 'train' or type == 'valid':
                itemlist = [path+type+'/part-'+str(x).zfill(5)+'.bz2' for x in range(0, n)]
        elif source == 's3':
            generator_to_use = BaseETL._read_file_generator_s3
            if type in ['train', 'valid']:
                itemlist = [type+'/part-'+str(x).zfill(5)+'.bz2' for x in range(0, n)]
            else:
                raise Exception('in create_itemlist_and_generator, got:\n\t{0}\n\t{1}\n\t{2}'\
                                .format(source, type, n))
        else:
            raise Exception('in create_itemlist_and_generator, got:\n{0}\n{1}\n{2}'\
                            .format(source, type, n))
        return (itemlist, generator_to_use)

    @staticmethod
    def create_base_rdd(sc, source, type, n):
        ''' creates the base_rdd that reads the files and maps out the fields 
            this is the one (and only) rdd created directly from the files
            :param sc:  sparkcontext
            :param source:  'file', 's3'
            :param type:    'mini', 'train', 'valid'  (mini is local only)
            :param n:   count  of files to load
            :returns:   base_rdd 
        '''
        itemlist, read_file_generator = BaseETL.create_itemlist_and_generator(source, type, n)
        rdd = sc.parallelize(itemlist) \
                .flatMap(read_file_generator) \
                .cache()
        return rdd

