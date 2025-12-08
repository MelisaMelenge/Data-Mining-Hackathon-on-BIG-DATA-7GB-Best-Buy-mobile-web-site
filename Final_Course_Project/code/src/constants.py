from datetime import datetime
import os

_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_fold = os.path.join(_root_dir, 'dataset')
out_buffer_path = os.path.join(_root_dir, 'output')
if not os.path.exists(out_buffer_path):
    os.makedirs(out_buffer_path)

train_file = os.path.join(data_fold, 'train.csv')
test_file = os.path.join(data_fold, 'test.csv')
out_file = os.path.join(out_buffer_path, 'prediction.csv')

new_train_file = os.path.join(data_fold, 'new_train.csv')
new_test_file = os.path.join(data_fold, 'new_test.csv')

st_date = datetime.strptime('2011-08-11 04:00:17', '%Y-%m-%d %H:%M:%S')
ed_date = datetime.strptime('2011-10-31 10:17:42', '%Y-%m-%d %H:%M:%S')

GLOBAL_QUERY = 6
GLOBAL_BIGRAM_QUERY = 6
w1 = 0.7
w2 = 0.3

duration = (ed_date - st_date).days
block_size = 12
MAX_BLOCK = block_size - 1
block = duration / block_size

PREDICT_HOT_SIZE = 'PREDICT_HOT_SIZE'
HOT = 'HOT'
BIGRAM_HOT = 'BIGRAM_HOT'
HOT_SIZE = 'HOT_SIZE'
SUM = 'SUM'
SUM_SIZE = 'SUM_SIZE'

magic_num = 100000

MAX_TEST_LINE = 1000 

