import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

standardlizator = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5,0.5,0.5),
                                                           std = (0.5,0.5,0.5))])

train_data = dsets.MNIST(root = 'data/', train=True, transform=standardlizator, download=True)
test_data = dsets.MNIST(root = 'data/', train=False, transform=standardlizator, download=True)

print('number of training data: ', len(train_data))
print('number of test data: ', len(test_data))

