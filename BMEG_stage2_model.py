import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from utils import readDependencyGraph


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class GCNNet(torch.nn.Module):
    def __init__(self, input_feat=1, n_micro=30):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(input_feat, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, n_micro)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_mean_pool(x, data.batch)  # Global pooling

        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        # print(x.shape)
        return self.softmax(x)


if __name__ == "__main__":
    BATCH_SIZE = 16
    model = GCNNet()
    x_data = torch.randn((30, 1))
    y_label = 2
    graph = torch.tensor(readDependencyGraph("/home/ad1238/bottleneck_detection_microservices/newExperiment/Data/realistic_aug9_25min_400_1/processed_traces/", "graph_paths_1"), dtype=torch.long)
    trainDataList = []
    edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
    for i in range(16):
        data = Data(x=x_data, edge_index = graph.t().contiguous(), y=y_label)
        trainDataList.append(data)

    trainloader = DataLoader(trainDataList, batch_size=512, shuffle=True)
    

    for i, data in enumerate(trainloader, 0):


        label = data.y
        # label = label
        #data = data.unsqueeze(dim=1)
        outputs = model(data)
        # print(label.shape, label.dtype)
        # print(outputs.shape, outputs.dtype)
        print(outputs.shape, label.shape)
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(outputs, label)
        # print(outputs)
        # print(label)
        
        # print(outputs.shape, label.shape)
        print(accuracy(outputs, label))
        print(label.unique(return_counts=True))