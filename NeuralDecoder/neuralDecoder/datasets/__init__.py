from .handwritingDataset import HandwritingDataset

def getDataset(datasetName):
    if datasetName == 'handwriting':
        return HandwritingDataset
    else:
        raise ValueError('Dataset not found')