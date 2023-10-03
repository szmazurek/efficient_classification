from torch import nn


class threeDClassModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(threeDClassModel, self).__init__()
        self.input_size = input_size

        self.conv_layer1 = self._conv_layer_set(input_size, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(175616, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.5)  # 0.15

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm3d(out_c),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        if out.shape[0] > 1:
            out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out
