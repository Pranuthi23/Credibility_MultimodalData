import sys 
import unittest
sys.path.append("./../")
import numpy as np 
from datasets import *
data_dir = "data"

class TestMultiModalDatasets(unittest.TestCase):
    
    train_dataloader, val_dataloader, test_dataloader = DATASET_DICT["mmimdb"](f"{data_dir}/mmimdb/multimodal_imdb.hdf5", batch_size=1, test_path=None, no_robust=True)
    n_train = 15552
    n_val = 18160 - 15552
    n_test = 25959 - 18160
    text_dim = 300
    image_dim = 256*160*3
    num_classes = 23 
    
    def test_dimensions(self):
        text, image, label = next(iter(self.train_dataloader))
        self.assertEqual(len(self.train_dataloader), self.n_train)
        self.assertEqual(len(self.val_dataloader), self.n_val)
        self.assertEqual(len(self.test_dataloader), self.n_test)
        self.assertEqual(np.prod(text.size()), self.text_dim)
        self.assertEqual(np.prod(image.size()), self.image_dim)
        self.assertEqual(np.prod(label.size()), self.num_classes)

if __name__ == '__main__':
    unittest.main()