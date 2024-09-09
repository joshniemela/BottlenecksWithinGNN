from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
import dataset as ds
from models import GCN

model = GCN(9, 32, 9, 4)
criterion = F.cross_entropy
optimizer = Adam(model.parameters(), lr=0.01)

trainData, testData = ds.train_test_split(ds.generate_tree(1000, 3))
trainData = DataLoader(trainData, batch_size=32)
testData = DataLoader(testData, batch_size=32)

model.train()

for epoch in range(50000):
    for data in trainData:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")
