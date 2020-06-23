# New Yorker caption contest -ml
Pre-processed New Yorker caption contest images and submitted captions for ML, including pytorch data loader.

## To use:

```
batch_size = 64
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
loader = get_loader('train', vocab, batch_size)
images, captions, lengths = next(iter(loader))
```

Data taken from [https://github.com/nextml/caption-contest-data](https://github.com/nextml/caption-contest-data)

Data loader based on [https://github.com/ajamjoom/Image-Captions](https://github.com/ajamjoom/Image-Captions)
