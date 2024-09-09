from torch.nn import functional as F
import dataset as ds
from models import GCN

criterion = F.cross_entropy

trainData, testData = ds.train_test_split(ds.generate_tree(1000, 3))

model = GCN(9, 32, 9)
model.train()
