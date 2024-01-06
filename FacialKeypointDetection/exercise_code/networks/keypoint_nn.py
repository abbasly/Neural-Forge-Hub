"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust the dimensions in case stride is not 1
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = F.elu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        out += self.shortcut(residual)
        out = F.elu(out)
        return out

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the fully connected layer
        # after the convolutional and pooling layers
        
        # Define fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 30)
    
        self.dropout = nn.Dropout(p=0.5)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.6)
        self.dropout7 = nn.Dropout(0.4)
        




        ########## TESTING ############
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.residual_block1 = ResidualBlock(16, 32, stride=1)
        # self.residual_block2 = ResidualBlock(32, 64, stride=1)
        # self.residual_block3 = ResidualBlock(64, 128, stride=1)
        # self.residual_block4 = ResidualBlock(128, 256, stride=1)

        # self.fc1 = nn.Linear(256*3*3, 128)
        # self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(256, 30)

        # self.dropout = nn.Dropout(p=0.5)


        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        

        # pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################


        x = F.elu(self.conv1(x))
        x = self.pool(x)
        # x = self.dropout1(x)

        x = F.elu(self.conv2(x))
        x = self.pool(x)
        # x = self.dropout2(x)

        x = F.elu(self.conv3(x))
        x = self.pool(x)
        # x = self.dropout3(x)

        x = F.elu(self.conv4(x))
        x = self.pool(x)
        # x = self.dropout4(x)

        x = F.elu(self.conv5(x))
        x = self.pool(x)

        # Flatten the input for the fully connected layers
        x = x.view(-1, 256 * 3 * 3)

        # Fully connected layers with ReLU activation and dropout
        x = F.elu(self.fc1(x))
        # x = self.dropout5(x)

        # x = self.fc2(x)
        x = F.elu(self.fc2(x))
        # x = self.dropout5(x)
        # x = nn.Identity()(self.fc2(x))
        x = self.fc3(x)


        ################ TEST ################

        # x = self.conv1(x)
        # # x = self.bn1(x)
        # x = F.elu(x)
        # x = self.pool(x)


        # x = self.residual_block1(x)

        # x = self.pool(x)

        # x = self.residual_block2(x)

        # x = self.pool(x)

        # x = self.residual_block3(x)
        
        # x = self.pool(x)

        # x = self.residual_block4(x)


        # x = self.pool(x)

        # # x = self.pool(x)

        # x = x.view(-1, 256*3*3)

        # x = F.elu(self.fc1(x))
        # x = F.elu(self.fc2(x))
        # x = self.fc3(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
