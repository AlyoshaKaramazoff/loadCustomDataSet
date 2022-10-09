import os
import torchvision
from matplotlib import pyplot as plt


# Сreate class to load train and test datasets
class loadDataSets:
    """ Класс для загрузки датасетов"""
    def __init__(self, PathToTrainset, PathToTestset):
        self.PathToTrainset = PathToTrainset
        self.PathToTestset = PathToTestset

    def getTrainset(self, message=False):
        trainset = torchvision.datasets.ImageFolder(self.PathToTrainset, transform=torchvision.transforms.ToTensor())
        if message:
            print('Size of training dataset :', len(trainset))
        return trainset

    def getTestset(self, message=False):
        testset = torchvision.datasets.ImageFolder(self.PathToTestset, transform=torchvision.transforms.ToTensor())
        if message:
            print('Size of test dataset :', len(testset))
        return testset

    def getClasses(self, message=False):
        classes = os.listdir(self.PathToTrainset)
        if message:
            print(len(classes), ' classes :', classes)
        return classes


# Create function for showing the image
def showImage(img, label, classes):
    print('Label: ', classes[label], '('+str(label)+')')
    plt.imshow(img.permute(1, 2, 0))
    plt.title(classes[label])
    plt.show()


if __name__ == "__main__":

    # Get the path to the dataset and the class names
    path = 'C:\\Users\\79110\\Desktop\\fruits-360_dataset\\fruits-360'
    pathToTrainset = path + '/Training'
    pathToTestset = path + '/Test'

    # Use the class loadDataSets, load Dataset and get trainset, testset and names of classes
    FruitsDataSet = loadDataSets(PathToTrainset=pathToTrainset,
                                 PathToTestset=pathToTestset)
    trainset = FruitsDataSet.getTrainset(message=False)
    testset = FruitsDataSet.getTestset(message=False)
    classes = FruitsDataSet.getClasses(message=False)

    # Get one image shape of the dataset.
    img, label = trainset[1000]

    # Show image
    showImage(img, label, classes)


