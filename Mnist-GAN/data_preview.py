from matplotlib import pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms

train_data = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_data  = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

image, label = train_data[0]

plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title('%i' % label)
plt.show()