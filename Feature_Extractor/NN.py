import torch


class NN(torch.nn.Module):
    """
    This is a neural network that takes fft, mfcc, and chroma data as input and extract features
    """

    def __init__(self, out_size):
        super(NN, self).__init__()
        conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv4 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_layers = torch.nn.Sequential(conv1, conv2, conv3, conv4)

        self.fc1 = torch.nn.Linear(256 * 2 * 2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, out_size)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

