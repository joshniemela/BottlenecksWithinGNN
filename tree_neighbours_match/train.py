from torch.nn import functional as F
from torch.optim import Adam
from torch import cuda
from torch_geometric.loader import DataLoader
import dataset as ds
from models import GCN

device = "cuda" if cuda.is_available() else "cpu"
device = "cpu"

model = GCN(9, 32, 9, 4).to(device)
criterion = F.cross_entropy
optimizer = Adam(model.parameters(), lr=0.01)

trainData, testData = ds.train_test_split(ds.generate_tree(1000, 3, device))
trainData = DataLoader(trainData, batch_size=516)
testData = DataLoader(testData, batch_size=32)

model.train()

for epoch in range(50000):
    for data in trainData:
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")
