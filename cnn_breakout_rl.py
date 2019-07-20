import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from skimage.transform import resize
# hyper params

gamma = 0.98

class Policy(nn.Module):

    def __init__(self):
        
        super(Policy, self).__init__()

        self.data = []
        self.lr = 0.002

        # define architecture/layer parameters
        self.input_channels = 3
        self.conv_ch_l1 = 8
        self.conv_ch_l2 = 12
        self.height = 210
        self.width = 160
        self.kernel_size = 3
        self.pool_size = 2

        self.conv_out = 23256
        self.fc1_size = 16
        self.fc_out = 4

        # deifne actual layer
        
        # define first convolutional layer
        self.conv1 = nn.Conv2d(in_channels = self.input_channels,
                                out_channels = self.conv_ch_l1,
                                kernel_size = self.kernel_size)

        # add batch normalization layer
        self.batch_norm1 = nn.BatchNorm2d(self.conv_ch_l1)

        # define max-pool layer
        self.pool = nn.MaxPool2d(self.pool_size, self.pool_size)

        # define second convolution layer
        self.conv2 = nn.Conv2d(in_channels = self.conv_ch_l1,
                                out_channels = self.conv_ch_l2,
                                kernel_size = self.kernel_size)

        # define batch normalization layer
        self.batch_norm2 = nn.BatchNorm2d(self.conv_ch_l2)


        # define fully connected layers
        self.fc1 = nn.Linear(self.conv_out, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc_out)

        # define optimizer
        self.optimizer = optim.Adam(self.parameters() , lr = self.lr)

    def forward(self, x):

        # pass input through conv layer
        out = self.pool(F.relu(self.conv1(x)))
        out = self.batch_norm1(out)

        out = self.pool(F.relu(self.conv2(out)))
        # print(out.size())
        # exit()
        out = self.batch_norm2(out)

        # reshape the conv out before passing it to fully connected layer
        _,b,c,d = out.size()
        fc_shape = b*c*d
        # print("FC input size : ", fc_shape)
        out = out.view(-1, fc_shape)

        # pass input through fully connected layer
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out

    
    # save data for training
    def put_data(self, item):

        self.data.append(item)

    # once the episode is complete we train the episode
    def train_policy(self):

        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + gamma * R
            loss = -log_prob * R

            # clean the previous gradients
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

        self.data = []


def main():

    # create the environment
    env = gym.make('Breakout-v0')
    pi = Policy()
    score = 0.0
    print_interval = 20
    num_episodes = 100

    for n_epi in range(num_episodes):
        state = env.reset()

        for t in range(100000):
            # state is an image with channel last.
            # pre-processing steps:
            # 1. make image grayscale
            # 2. resize image
            # 3. add first dimension for batch 
            # 4. convert image to tensor

            #img = np.dot(state[:,:,:3], [0.2989, 0.5870, 0.1140])
            #img = resize(img, (63, 48), anti_aliasing=True)
            # now image is converted to single channel, add dimension for channel
            #img = np.expand_dims(img, axis=0)
            img = np.rollaxis(state, 2, 0)  
            prob = pi(torch.from_numpy(img).unsqueeze(0).float())

            m = Categorical(prob)
            a = m.sample()

            state_prime, r, done, _ = env.step(a.item())
            # print(prob.size())
            # print(prob)
            # print(a)
            # print(a.size())
            # exit()
            print("Output : ", prob)
            print("Action : ", a.item())
            print("Reward : ", r)
            pi.put_data((r,torch.log(prob[0,a])))

            state = state_prime
            score += r
            if done:
                print("Episode ended : ", n_epi+1)
                break

            # if the episode is completed, train policy on recorded observations
            pi.train_policy()

        if (n_epi+1)%print_interval == 0 and n_epi > 0 :
            print("Episode : {}, avg_score : {}".format(n_epi, 
                                                        score/print_interval)
                )
            score = 0

        env.close()

if __name__ == '__main__':
    main()
                




