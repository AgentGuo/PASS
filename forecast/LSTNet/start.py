import os
from data_preprocessing import *
from tojson import *
from rawdata_to_npz import *



cmd = "time python main.py --data ./data/hybrid_101.npz --save_pb ./save_pb/hybrid_101  --save ./save/hybrid_101.pk  --log ./logs/hybrid_101.log --save_weights ./save_weights/hybrid_101 " \
      "--exps 1 --patience 15 --normalize 1 --loss mae --hidCNN 100 --hidRNN 100 --hidSkip 50 --output_fun linear --multi 1 --add_model 0 --horizon 3  --highway_window 7 --window 14 --skip 7 --ps 3"

os.system(cmd)

