# New Yorker caption contest dataset and loader

!![Funny caption](https://raw.githubusercontent.com/benlansdell/ny-cartoon-ml/master/508.jpg?token=ABZIQN42ZWZQDDLGFGWHLK267NXOW "Logo Title Text 1")

Pre-processed New Yorker caption contest images and submitted captions for ML, including pytorch data loader.

187 training images and 39 validation images, along with 926195 corresponding training captions and 99298 validation captions.

Data have been resized to 224x224 jpgs. 

Encodes text data using a vocabulary built from the COCO image annotations dataset. Pre-built in vocab.pkl. You can modify `prepare_vocab.py` to change this.

## To use:

```
batch_size = 64
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
train_loader = get_loader('train', vocab, batch_size)
val_loader = get_loader('train', vocab, batch_size)

images, captions, lengths = next(iter(train_loader))
```

Raw data obtained from [https://github.com/nextml/caption-contest-data](https://github.com/nextml/caption-contest-data)

Data loader based on [https://github.com/ajamjoom/Image-Captions](https://github.com/ajamjoom/Image-Captions)
