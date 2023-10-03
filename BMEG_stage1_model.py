import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool, global_add_pool
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

def readDependencyGraph(PATH, fileName):
    file = open(PATH+fileName, 'r')
    lines = file.readlines()
    graph = []
    for line in lines:
        line = line.strip()
        edges = []
        nodes = line.split("->")
        #print(nodes)
        for i in range(len(nodes)-1):
            graph.append([int(nodes[i]), int(nodes[i+1])])
            # graph.append([int(nodes[i+1])-1, int(nodes[i])-1])
    return graph


# Define the DGCNN architecture
class DGCNN(torch.nn.Module):
    def __init__(self, input_feat = 1, output_feat=2):
        super(DGCNN, self).__init__()
        
        self.conv1 = GCNConv(input_feat, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = GCNConv(32, 32)
        self.conv6 = GCNConv(32, 32)
        self.conv7 = GCNConv(32, 32)

        # Post-processing of the graph embedding
        self.fc1 = torch.nn.Linear(192, 128)
        self.fc2 = torch.nn.Linear(128, output_feat)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # print(x.shape)
        # x = x.float()
        x = F.relu(self.conv1(x, edge_index))
        x1 = global_add_pool(x, batch)

        x = F.relu(self.conv2(x, edge_index))
        x2 = global_add_pool(x, batch)

        x = F.relu(self.conv3(x, edge_index))
        x3 = global_add_pool(x, batch)

        x = F.relu(self.conv4(x, edge_index))
        x4 = global_add_pool(x, batch)

        x = F.relu(self.conv5(x, edge_index))
        x5 = global_add_pool(x, batch)

        x = F.relu(self.conv6(x, edge_index))
        x6 = global_add_pool(x, batch)

        # Concatenate and pass through MLP
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)

        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)

        return self.softmax(x)


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


if __name__ == "__main__":
    BATCH_SIZE = 16
    model = DGCNN().to('cuda')
    temp = torch.randn((30, 1))
    graph = torch.tensor(readDependencyGraph("/home/ad1238/bottleneck_detection_microservices/newExperiment/Data/realistic_aug9_25min_400_1/processed_traces/", "graph_paths_1"), dtype=torch.long)
    trainDataList = []
    edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
    for i in range(16):
        data = Data(x=temp, edge_index = graph.t().contiguous(), y=1)
        trainDataList.append(data)

    # trainloader = DataLoader(trainDataList, batch_size=BATCH_SIZE, shuffle=True)

    
    
    train_dataset = torch.load("./BMEG_dataset/test_BMEG.pt")
    
    
    trainloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    

    for i, data in enumerate(trainloader, 0):


        label = data.y
        label = label.to('cuda')
        #data = data.unsqueeze(dim=1)
        outputs = model(data.to('cuda'))
        # print(label.shape, label.dtype)
        # print(outputs.shape, outputs.dtype)
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(outputs, label)
        # print(outputs)
        # print(label)
        
        # print(outputs.shape, label.shape)
        print(accuracy(outputs, label))
        print(label.unique(return_counts=True))


