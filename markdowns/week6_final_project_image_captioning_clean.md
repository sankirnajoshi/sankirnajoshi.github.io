<a href="https://colab.research.google.com/github/sankirnajoshi/intro-to-dl/blob/master/week6/week6_final_project_image_captioning_clean.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
! shred -u setup_google_colab.py
! wget https://raw.githubusercontent.com/hse-aml/intro-to-dl/master/setup_google_colab.py -O setup_google_colab.py
import setup_google_colab
# please, uncomment the week you're working on
# setup_google_colab.setup_week1()
# setup_google_colab.setup_week2()
# setup_google_colab.setup_week2_honor()
# setup_google_colab.setup_week3()
# setup_google_colab.setup_week4()
# setup_google_colab.setup_week5()
setup_google_colab.setup_week6()
```

    shred: setup_google_colab.py: failed to open for writing: No such file or directory
    --2020-01-11 16:35:03--  https://raw.githubusercontent.com/hse-aml/intro-to-dl/master/setup_google_colab.py
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3636 (3.6K) [text/plain]
    Saving to: ‘setup_google_colab.py’
    
    setup_google_colab. 100%[===================>]   3.55K  --.-KB/s    in 0s      
    
    2020-01-11 16:35:03 (94.3 MB/s) - ‘setup_google_colab.py’ saved [3636/3636]
    
    **************************************************
    captions_train-val2014.zip
    **************************************************
    train2014_sample.zip
    **************************************************
    train_img_embeds.pickle
    **************************************************
    train_img_fns.pickle
    **************************************************
    val2014_sample.zip
    **************************************************
    val_img_embeds.pickle
    **************************************************
    val_img_fns.pickle
    **************************************************
    inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    **************************************************
    cifar-10-batches-py.tar.gz
    **************************************************
    mnist.npz
    


```python
# set tf 1.x for colab
%tensorflow_version 1.x
```

# Image Captioning Final Project

In this final project you will define and train an image-to-caption model, that can produce descriptions for real world images!

<img src="https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/encoder_decoder.png?raw=1" style="width:70%">

Model architecture: CNN encoder and RNN decoder. 
(https://research.googleblog.com/2014/11/a-picture-is-worth-thousand-coherent.html)

# Import stuff


```python
import sys
sys.path.append("..")
import grading
import download_utils
```


```python
download_utils.link_all_keras_resources()
```


```python
import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
L = keras.layers
K = keras.backend
import utils
import time
import zipfile
import json
from collections import defaultdict
import re
import random
from random import choice
import grading_utils
import os
from keras_utils import reset_tf_session
import tqdm_utils
```

    Using TensorFlow backend.
    

# Prepare the storage for model checkpoints


```python
# Leave USE_GOOGLE_DRIVE = False if you're running locally!
# We recommend to set USE_GOOGLE_DRIVE = True in Google Colab!
# If set to True, we will mount Google Drive, so that you can restore your checkpoint 
# and continue trainig even if your previous Colab session dies.
# If set to True, follow on-screen instructions to access Google Drive (you must have a Google account).
USE_GOOGLE_DRIVE = True

def mount_google_drive():
    from google.colab import drive
    mount_directory = "/content/gdrive"
    drive.mount(mount_directory)
    drive_root = mount_directory + "/" + list(filter(lambda x: x[0] != '.', os.listdir(mount_directory)))[0] + "/colab"
    return drive_root

CHECKPOINT_ROOT = ""
if USE_GOOGLE_DRIVE:
    CHECKPOINT_ROOT = mount_google_drive() + "/"

def get_checkpoint_path(epoch=None):
    if epoch is None:
        return os.path.abspath(CHECKPOINT_ROOT + "weights")
    else:
        return os.path.abspath(CHECKPOINT_ROOT + "weights_{}".format(epoch))
      
# example of checkpoint dir
print(get_checkpoint_path(10))
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).
    /content/gdrive/My Drive/colab/weights_10
    

# Fill in your Coursera token and email
To successfully submit your answers to our grader, please fill in your Coursera submission token and email


```python
grader = grading.Grader(assignment_key="NEDBg6CgEee8nQ6uE8a7OA", 
                        all_parts=["19Wpv", "uJh73", "yiJkt", "rbpnH", "E2OIL", "YJR7z"])
```


```python
# token expires every 30 min
#COURSERA_TOKEN = ### YOUR TOKEN HERE
COURSERA_EMAIL = "sankirna1292@gmail.com"
```

# Download data

Takes 10 hours and 20 GB. We've downloaded necessary files for you.

Relevant links (just in case):
- train images http://msvocds.blob.core.windows.net/coco2014/train2014.zip
- validation images http://msvocds.blob.core.windows.net/coco2014/val2014.zip
- captions for both train and validation http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip


```python
# we downloaded them for you, just link them here
download_utils.link_week_6_resources()
```

# Extract image features

We will use pre-trained InceptionV3 model for CNN encoder (https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html) and extract its last hidden layer as an embedding:

<img src="https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/inceptionv3.png?raw=1" style="width:70%">


```python
IMG_SIZE = 299
```


```python
# we take the last hidden layer of IncetionV3 as an image embedding
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model
```

Features extraction takes too much time on CPU:
- Takes 16 minutes on GPU.
- 25x slower (InceptionV3) on CPU and takes 7 hours.
- 10x slower (MobileNet) on CPU and takes 3 hours.

So we've done it for you with the following code:
```python
# load pre-trained model
reset_tf_session()
encoder, preprocess_for_model = get_cnn_encoder()

# extract train features
train_img_embeds, train_img_fns = utils.apply_model(
    "train2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
utils.save_pickle(train_img_embeds, "train_img_embeds.pickle")
utils.save_pickle(train_img_fns, "train_img_fns.pickle")

# extract validation features
val_img_embeds, val_img_fns = utils.apply_model(
    "val2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
utils.save_pickle(val_img_embeds, "val_img_embeds.pickle")
utils.save_pickle(val_img_fns, "val_img_fns.pickle")

# sample images for learners
def sample_zip(fn_in, fn_out, rate=0.01, seed=42):
    np.random.seed(seed)
    with zipfile.ZipFile(fn_in) as fin, zipfile.ZipFile(fn_out, "w") as fout:
        sampled = filter(lambda _: np.random.rand() < rate, fin.filelist)
        for zInfo in sampled:
            fout.writestr(zInfo, fin.read(zInfo))
            
sample_zip("train2014.zip", "train2014_sample.zip")
sample_zip("val2014.zip", "val2014_sample.zip")
```


```python
# load prepared embeddings
train_img_embeds = utils.read_pickle("train_img_embeds.pickle")
train_img_fns = utils.read_pickle("train_img_fns.pickle")
val_img_embeds = utils.read_pickle("val_img_embeds.pickle")
val_img_fns = utils.read_pickle("val_img_fns.pickle")
# check shapes
print(train_img_embeds.shape, len(train_img_fns))
print(val_img_embeds.shape, len(val_img_fns))
```

    (82783, 2048) 82783
    (40504, 2048) 40504
    


```python
# check prepared samples of images
list(filter(lambda x: x.endswith("_sample.zip"), os.listdir(".")))
```




    ['val2014_sample.zip', 'train2014_sample.zip']



# Extract captions for images


```python
# extract captions from zip
def get_captions_for_fns(fns, zip_fn, zip_json_path):
    zf = zipfile.ZipFile(zip_fn)
    j = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_fn = {img["id"]: img["file_name"] for img in j["images"]}
    fn_to_caps = defaultdict(list)
    for cap in j['annotations']:
        fn_to_caps[id_to_fn[cap['image_id']]].append(cap['caption'])
    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))
    
train_captions = get_captions_for_fns(train_img_fns, "captions_train-val2014.zip", 
                                      "annotations/captions_train2014.json")

val_captions = get_captions_for_fns(val_img_fns, "captions_train-val2014.zip", 
                                      "annotations/captions_val2014.json")

# check shape
print(len(train_img_fns), len(train_captions))
print(len(val_img_fns), len(val_captions))
```

    82783 82783
    40504 40504
    


```python
# look at training example (each has 5 captions)
def show_trainig_example(train_img_fns, train_captions, example_idx=0):
    """
    You can change example_idx and see different images
    """
    zf = zipfile.ZipFile("train2014_sample.zip")
    captions_by_file = dict(zip(train_img_fns, train_captions))
    all_files = set(train_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    img = utils.decode_image_from_buf(zf.read(example))
    plt.imshow(utils.image_center_crop(img))
    plt.title("\n".join(captions_by_file[example.filename.rsplit("/")[-1]]))
    plt.show()
    
show_trainig_example(train_img_fns, train_captions, example_idx=33)
```


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_23_0.png)


# Prepare captions for training


```python
# preview captions data
train_captions[:2]
```




    [['A long dirt road going through a forest.',
      'A SCENE OF WATER AND A PATH WAY',
      'A sandy path surrounded by trees leads to a beach.',
      'Ocean view through a dirt road surrounded by a forested area. ',
      'dirt path leading beneath barren trees to open plains'],
     ['A group of zebra standing next to each other.',
      'This is an image of of zebras drinking',
      'ZEBRAS AND BIRDS SHARING THE SAME WATERING HOLE',
      'Zebras that are bent over and drinking water together.',
      'a number of zebras drinking water near one another']]




```python
# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

# split sentence into tokens (split into lowercased words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))

def generate_vocabulary(train_captions):
    """
    Return {token: index} for all train tokens (words) that occur 5 times or more, 
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    Use `split_sentence` function to split sentence into tokens.
    Also, add PAD (for batch padding), UNK (unknown, out of vocabulary), 
        START (start of sentence) and END (end of sentence) tokens into the vocabulary.
    """

    words = []
    from collections import Counter
    for captions in train_captions: #multiple captions for an image
        for caption in captions: #for every caption in the captions
            words.extend(split_sentence(caption)) #take every word every caption
    words.extend([PAD,UNK,START,END]) 
    A = Counter(words) #removes duplicates and gives repetitions. We filter cases that occur less than five.
    vocab_counter = {k:v for (k,v) in dict(A).items() if v >= 5 or k in (PAD,UNK,START,END)}
    vocab = vocab_counter.keys() #Storing all words just in case if required ahead

    return {token: index for index, token in enumerate(sorted(vocab))}  

def caption_tokens_to_indices(captions, vocab):
    """
    `captions` argument is an array of arrays:
    [
        [
            "image1 caption1",
            "image1 caption2",
            ...
        ],
        [
            "image2 caption1",
            "image2 caption2",
            ...
        ],
        ...
    ]
    Use `split_sentence` function to split sentence into tokens.
    Replace all tokens with vocabulary indices, use UNK for unknown words (out of vocabulary).
    Add START and END tokens to start and end of each sentence respectively.
    For the example above you should produce the following:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    """
    lambda_words = lambda words: [vocab[START]] + [vocab[w] if w in vocab else vocab[UNK] for w in words] + [vocab[END]]
    
    lambda_sentences = lambda captions: [lambda_words(split_sentence(caption)) for caption in captions]
    res = [lambda_sentences(caption_list) for caption_list in captions]
       
    return res
```


```python
# prepare vocabulary
vocab = generate_vocabulary(train_captions)
vocab_inverse = {idx: w for w, idx in vocab.items()}
print(len(vocab))
```

    8769
    


```python
# replace tokens with indices
train_captions_indexed = caption_tokens_to_indices(train_captions, vocab)
val_captions_indexed = caption_tokens_to_indices(val_captions, vocab)
```

Captions have different length, but we need to batch them, that's why we will add PAD tokens so that all sentences have an equal length. 

We will crunch LSTM through all the tokens, but we will ignore padding tokens during loss calculation.


```python
# we will use this during training
def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    """
    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    Put vocabulary indexed captions into np.array of shape (len(batch_captions), columns),
        where "columns" is max(map(len, batch_captions)) when max_len is None
        and "columns" = min(max_len, max(map(len, batch_captions))) otherwise.
    Add padding with pad_idx where necessary.
    Input example: [[1, 2, 3], [4, 5]]
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
    Output example: np.array([[1, 2], [4, 5]]) if max_len=2
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
    Try to use numpy, we need this function to be fast!
    """
    
    max_len = max_len or max(map(len, batch_captions)) #if None, then max(map(len, batch_captions))
    max_len = min(max_len, max(map(len, batch_captions))) #otherwise
    matrix = np.empty((len(batch_captions), max_len)) #required shape
    matrix.fill(pad_idx)
    for i in range(len(batch_captions)):
        line_ix = list(batch_captions[i])[:max_len]
        matrix[i,:len(line_ix)] = line_ix
    return matrix
```


```python
## GRADED PART, DO NOT CHANGE!
# Vocabulary creation
grader.set_answer("19Wpv", grading_utils.test_vocab(vocab, PAD, UNK, START, END))
# Captions indexing
grader.set_answer("uJh73", grading_utils.test_captions_indexing(train_captions_indexed, vocab, UNK))
# Captions batching
grader.set_answer("yiJkt", grading_utils.test_captions_batching(batch_captions_to_matrix))
```


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit("sankirna1292@gmail.com", "zTGcnIroHlGoJQfb")
```

    You used an invalid email or your token may have expired. Please make sure you have entered all fields correctly. Try generating a new token if the issue still persists.
    


```python
# make sure you use correct argument in caption_tokens_to_indices
assert len(caption_tokens_to_indices(train_captions[:10], vocab)) == 10
assert len(caption_tokens_to_indices(train_captions[:5], vocab)) == 5
```

# Training

## Define architecture

Since our problem is to generate image captions, RNN text generator should be conditioned on image. The idea is to use image features as an initial state for RNN instead of zeros. 

Remember that you should transform image feature vector to RNN hidden state size by fully-connected layer and then pass it to RNN.

During training we will feed ground truth tokens into the lstm to get predictions of next tokens. 

Notice that we don't need to feed last token (END) as input (http://cs.stanford.edu/people/karpathy/):

<img src="https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/encoder_decoder_explained.png?raw=1" style="width:50%">


```python
IMG_EMBED_SIZE = train_img_embeds.shape[1]
print(IMG_EMBED_SIZE)
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
pad_idx = vocab[PAD]
```

    2048
    


```python
# remember to reset your graph if you want to start building it from scratch!
s = reset_tf_session()
tf.set_random_seed(42)
```

Here we define decoder graph.

We use Keras layers where possible because we can use them in functional style with weights reuse like this:
```python
dense_layer = L.Dense(42, input_shape=(None, 100) activation='relu')
a = tf.placeholder('float32', [None, 100])
b = tf.placeholder('float32', [None, 100])
dense_layer(a)  # that's how we applied dense layer!
dense_layer(b)  # and again
```

![alt text](https://)Here's a figure to help you with flattening in decoder:
<img src="https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/flatten_help.jpg?raw=1" style="width:80%">


```python
class decoder:
    # [batch_size, IMG_EMBED_SIZE] of CNN image features
    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])
    # [batch_size, time steps] of word ids
    sentences = tf.placeholder('int32', [None, None])
    
    # we use bottleneck here to reduce the number of parameters
    # image embedding -> bottleneck
    img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK, 
                                      input_shape=(None, IMG_EMBED_SIZE), 
                                      activation='elu')
    # image embedding bottleneck -> lstm initial state
    img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')
    # word -> embedding
    word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)
    # lstm cell (from tensorflow)
    lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
    
    # we use bottleneck here to reduce model complexity
    # lstm output -> logits bottleneck
    token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK, 
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")
    # logits bottleneck -> logits for next token prediction
    token_logits = L.Dense(len(vocab),
                           input_shape=(None, LOGIT_BOTTLENECK))
    
    # initial lstm cell state of shape (None, LSTM_UNITS),
    # we need to condition it on `img_embeds` placeholder.
    c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds)) ### YOUR CODE HERE ###

    # embed all tokens but the last for lstm input,
    # remember that L.Embedding is callable,
    # use `sentences` placeholder as input.
    
    word_embeds = word_embed(sentences[:, :-1]) ### YOUR CODE HERE ###

    
    # during training we use ground truth tokens `word_embeds` as context for next token prediction.
    # that means that we know all the inputs for our lstm and can get 
    # all the hidden states with one tensorflow operation (tf.nn.dynamic_rnn).
    # `hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].
    hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,
                                         initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

    # now we need to calculate token logits for all the hidden states
    
    # first, we reshape `hidden_states` to [-1, LSTM_UNITS]
    flat_hidden_states = tf.reshape(hidden_states, [-1, LSTM_UNITS]) ### YOUR CODE HERE ###

    # then, we calculate logits for next tokens using `token_logits_bottleneck` and `token_logits` layers
    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states)) ### YOUR CODE HERE ###
    
    # then, we flatten the ground truth token ids.
    # remember, that we predict next tokens for each time step,
    # use `sentences` placeholder.
    flat_ground_truth = tf.reshape(sentences[:, 1:], [-1])### YOUR CODE HERE ###

    # we need to know where we have real tokens (not padding) in `flat_ground_truth`,
    # we don't want to propagate the loss for padded output tokens,
    # fill `flat_loss_mask` with 1.0 for real tokens (not pad_idx) and 0.0 otherwise.
    flat_loss_mask = tf.not_equal(pad_idx, flat_ground_truth) ### YOUR CODE HERE ###
    flat_loss_mask = tf.cast(flat_loss_mask, tf.float32) #As tf.not_equal produces a bool type tensor resulting in a TypeError : Input 'y' of 'Mul' Op has type bool that does not match type float32 of argument 'x'.

    # compute cross-entropy between `flat_ground_truth` and `flat_token_logits` predicted by lstm
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_ground_truth, 
        logits=flat_token_logits
    )

    # compute average `xent` over tokens with nonzero `flat_loss_mask`.
    # we don't want to account misclassification of PAD tokens, because that doesn't make sense,
    # we have PAD tokens for batching purposes only!
    loss = tf.reduce_mean(tf.boolean_mask(xent,flat_loss_mask)) ### YOUR CODE HERE ###
```


```python

```


```python
# define optimizer operation to minimize the loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(decoder.loss)

# will be used to save/load network weights.
# you need to reset your default graph and define it in the same way to be able to load the saved weights!
saver = tf.train.Saver()

# intialize all variables
s.run(tf.global_variables_initializer())
```


```python
## GRADED PART, DO NOT CHANGE!
# Decoder shapes test
grader.set_answer("rbpnH", grading_utils.test_decoder_shapes(decoder, IMG_EMBED_SIZE, vocab, s))
# Decoder random loss test
grader.set_answer("E2OIL", grading_utils.test_random_decoder_loss(decoder, IMG_EMBED_SIZE, vocab, s))
```


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit("sankirna1292@gmail.com", "Z3NabLzgjxBTkzPE")
```

    You used an invalid email or your token may have expired. Please make sure you have entered all fields correctly. Try generating a new token if the issue still persists.
    

## Training loop
Evaluate train and validation metrics through training and log them. Ensure that loss decreases.


```python
train_captions_indexed = np.array(train_captions_indexed)
val_captions_indexed = np.array(val_captions_indexed)
```


```python
# generate batch via random sampling of images and captions for them,
# we use `max_len` parameter to control the length of the captions (truncating long captions)
def generate_batch(images_embeddings, indexed_captions, batch_size, max_len=None):
    """
    `images_embeddings` is a np.array of shape [number of images, IMG_EMBED_SIZE].
    `indexed_captions` holds 5 vocabulary indexed captions for each image:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    Generate a random batch of size `batch_size`.
    Take random images and choose one random caption for each image.
    Remember to use `batch_captions_to_matrix` for padding and respect `max_len` parameter.
    Return feed dict {decoder.img_embeds: ..., decoder.sentences: ...}.
    """
    indices = np.random.choice(images_embeddings.shape[0], batch_size, replace=False)
    batch_image_embeddings = images_embeddings[indices]  ### YOUR CODE HERE ###
    
    caption_indices =  np.random.choice(5, batch_size)
    batch_captions_matrix = [indexed_captions[indx][cap] for indx, cap in zip(indices, caption_indices)]
    batch_captions_matrix = batch_captions_to_matrix(batch_captions_matrix, pad_idx) ### YOUR CODE HERE ###
    
    return {decoder.img_embeds: batch_image_embeddings, 
            decoder.sentences: batch_captions_matrix}
```


```python
batch_size = 64
n_epochs = 11
n_batches_per_epoch = 1000
n_validation_batches = 100  # how many batches are used for validation after each epoch
```


```python
# you can load trained weights here
# uncomment the next line if you need to load weights
#saver.restore(s, get_checkpoint_path(epoch=6))
```

Look at the training and validation loss, they should be decreasing!


```python
# actual training loop
MAX_LEN = 20  # truncate long captions to speed up training

# to make training reproducible
np.random.seed(42)
random.seed(42)

for epoch in range(n_epochs):
    
    train_loss = 0
    pbar = tqdm_utils.tqdm_notebook_failsafe(range(n_batches_per_epoch))
    counter = 0
    for _ in pbar:
        train_loss += s.run([decoder.loss, train_step], 
                            generate_batch(train_img_embeds, 
                                           train_captions_indexed, 
                                           batch_size, 
                                           MAX_LEN))[0]
        counter += 1
        pbar.set_description("Training loss: %f" % (train_loss / counter))
        
    train_loss /= n_batches_per_epoch
    
    val_loss = 0
    for _ in range(n_validation_batches):
        val_loss += s.run(decoder.loss, generate_batch(val_img_embeds,
                                                       val_captions_indexed, 
                                                       batch_size, 
                                                       MAX_LEN))
    val_loss /= n_validation_batches
    
    print('Epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

    # save weights after finishing epoch
    saver.save(s, get_checkpoint_path(epoch))
    
print("Finished!")
```

    **************************************************
    Training loss: 2.546536
    Epoch: 0, train loss: 2.5465357267856596, val loss: 2.6588618969917297
    **************************************************
    Training loss: 2.514132
    Epoch: 1, train loss: 2.5141322622299196, val loss: 2.600285520553589
    **************************************************
    Training loss: 2.501045
    Epoch: 2, train loss: 2.501045058250427, val loss: 2.6063569498062136
    **************************************************
    Training loss: 2.469324
    Epoch: 3, train loss: 2.469323918104172, val loss: 2.6032228565216062
    **************************************************
    Training loss: 2.455755
    Epoch: 4, train loss: 2.455755119562149, val loss: 2.603102903366089
    **************************************************
    Training loss: 2.440631
    Epoch: 5, train loss: 2.440631441116333, val loss: 2.57949627161026
    **************************************************
    Training loss: 2.418618
    Epoch: 6, train loss: 2.4186184751987456, val loss: 2.5686046195030214
    **************************************************
    Training loss: 2.402204
    Epoch: 7, train loss: 2.402203711986542, val loss: 2.574491784572601
    **************************************************
    Training loss: 2.393621
    Epoch: 8, train loss: 2.393621375322342, val loss: 2.544399118423462
    **************************************************
    Training loss: 2.402825
    Epoch: 9, train loss: 2.402824951171875, val loss: 2.549779553413391
    **************************************************
    Training loss: 2.387232
    Epoch: 10, train loss: 2.3872317142486574, val loss: 2.5369355845451356
    Finished!
    


```python
## GRADED PART, DO NOT CHANGE!
# Validation loss
grader.set_answer("YJR7z", grading_utils.test_validation_loss(
    decoder, s, generate_batch, val_img_embeds, val_captions_indexed))
```

    **************************************************
    
    


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, "ktfnpggs06g6qTbW")
```

    Submitted to Coursera platform. See results on assignment page!
    


```python
# check that it's learnt something, outputs accuracy of next word prediction (should be around 0.5)
from sklearn.metrics import accuracy_score, log_loss

def decode_sentence(sentence_indices):
    return " ".join(list(map(vocab_inverse.get, sentence_indices)))

def check_after_training(n_examples):
    fd = generate_batch(train_img_embeds, train_captions_indexed, batch_size)
    logits = decoder.flat_token_logits.eval(fd)
    truth = decoder.flat_ground_truth.eval(fd)
    mask = decoder.flat_loss_mask.eval(fd).astype(bool)
    print("Loss:", decoder.loss.eval(fd))
    print("Accuracy:", accuracy_score(logits.argmax(axis=1)[mask], truth[mask]))
    for example_idx in range(n_examples):
        print("Example", example_idx)
        print("Predicted:", decode_sentence(logits.argmax(axis=1).reshape((batch_size, -1))[example_idx]))
        print("Truth:", decode_sentence(truth.reshape((batch_size, -1))[example_idx]))
        print("")

check_after_training(3)
```

    Loss: 2.4177282
    Accuracy: 0.49343832020997375
    Example 0
    Predicted: a man and a into a of a #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END#
    Truth: a man pouring wine from #UNK# for patrons #END# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD#
    
    Example 1
    Predicted: a man and to of banana to on #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END#
    Truth: a man tries out a bicycle powered blender #END# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD#
    
    Example 2
    Predicted: a toilet up of a toilet toilet toilet #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END#
    Truth: a close up of a clean white toilet #END# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD#
    
    


```python
# save last graph weights to file!
saver.save(s, get_checkpoint_path())
```




    '/content/gdrive/My Drive/colab/weights'



# Applying model

Here we construct a graph for our final model.

It will work as follows:
- take an image as an input and embed it
- condition lstm on that embedding
- predict the next token given a START input token
- use predicted token as an input at next time step
- iterate until you predict an END token


```python
class final_model:
    # CNN encoder
    encoder, preprocess_for_model = get_cnn_encoder()
    saver.restore(s, get_checkpoint_path())  # keras applications corrupt our graph, so we restore trained weights
    
    # containers for current lstm state
    lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="cell")
    lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="hidden")

    # input images
    input_images = tf.placeholder('float32', [1, IMG_SIZE, IMG_SIZE, 3], name='images')

    # get image embeddings
    img_embeds = encoder(input_images)

    # initialize lstm state conditioned on image
    init_c = init_h = decoder.img_embed_bottleneck_to_h0(decoder.img_embed_to_bottleneck(img_embeds))
    init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)
    
    # current word index
    current_word = tf.placeholder('int32', [1], name='current_input')

    # embedding for current word
    word_embed = decoder.word_embed(current_word)

    # apply lstm cell, get new lstm states
    new_c, new_h = decoder.lstm(word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]

    # compute logits for next token
    new_logits = decoder.token_logits(decoder.token_logits_bottleneck(new_h))
    # compute probabilities for next token
    new_probs = tf.nn.softmax(new_logits)

    # `one_step` outputs probabilities of next token and updates lstm hidden state
    one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)
```

    INFO:tensorflow:Restoring parameters from /content/gdrive/My Drive/colab/weights
    


```python
# look at how temperature works for probability distributions
# for high temperature we have more uniform distribution
_ = np.array([0.5, 0.4, 0.1])
for t in [0.01, 0.1, 1, 10, 100]:
    print(" ".join(map(str, _**(1/t) / np.sum(_**(1/t)))), "with temperature", t)
```

    0.9999999997962965 2.0370359759195462e-10 1.2676505999700117e-70 with temperature 0.01
    0.9030370433250645 0.09696286420394223 9.247099323648666e-08 with temperature 0.1
    0.5 0.4 0.1 with temperature 1
    0.35344772639219624 0.34564811360592396 0.3009041600018798 with temperature 10
    0.33536728048099185 0.33461976434857876 0.3300129551704294 with temperature 100
    


```python
# this is an actual prediction loop
def generate_caption(image, t=1, sample=False, max_len=20):
    """
    Generate caption for given image.
    if `sample` is True, we will sample next token from predicted probability distribution.
    `t` is a temperature during that sampling,
        higher `t` causes more uniform-like distribution = more chaos.
    """
    # condition lstm on the image
    s.run(final_model.init_lstm, 
          {final_model.input_images: [image]})
    
    # current caption
    # start with only START token
    caption = [vocab[START]]
    
    for _ in range(max_len):
        next_word_probs = s.run(final_model.one_step, 
                                {final_model.current_word: [caption[-1]]})[0]
        next_word_probs = next_word_probs.ravel()
        
        # apply temperature
        next_word_probs = next_word_probs**(1/t) / np.sum(next_word_probs**(1/t))

        if sample:
            next_word = np.random.choice(range(len(vocab)), p=next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == vocab[END]:
            break
       
    return list(map(vocab_inverse.get, caption))
```


```python
# look at validation prediction example
def apply_model_to_image_raw_bytes(raw):
    img = utils.decode_image_from_buf(raw)
    fig = plt.figure(figsize=(7, 7))
    plt.grid('off')
    plt.axis('off')
    plt.imshow(img)
    img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model)
    print(' '.join(generate_caption(img)[1:-1]))
    plt.show()

def show_valid_example(val_img_fns, example_idx=0):
    zf = zipfile.ZipFile("val2014_sample.zip")
    all_files = set(val_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    apply_model_to_image_raw_bytes(zf.read(example))
    
show_valid_example(val_img_fns, example_idx=100)
```

    a baseball player swinging a bat at a game
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_61_1.png)



```python
# sample more images from validation
for idx in np.random.choice(range(len(zipfile.ZipFile("val2014_sample.zip").filelist) - 1), 10):
    show_valid_example(val_img_fns, example_idx=idx)
    time.sleep(1)
```

    a man holding a knife and a knife on a cutting board
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_1.png)


    a sheep standing next to a fence in a field
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_3.png)


    a street sign with a sign on it
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_5.png)


    a bowl of fruit with apples and oranges
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_7.png)


    a man and woman are standing in the grass
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_9.png)


    a kitchen with a stove top oven and a stove top oven
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_11.png)


    a double decker bus is parked on the side of the road
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_13.png)


    a room with a bed and a chair in it
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_15.png)


    a bicycle is parked on the side of a street
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_17.png)


    a man swinging a tennis racquet on a tennis court
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_62_19.png)


You can download any image from the Internet and appply your model to it!


```python
download_utils.download_file(
    #"https://www.bijouxandbits.com/wp-content/uploads/2016/06/portal-cake-10.jpg","portal-cake-10.jpg"
    "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AABHnbv.img?h=552&w=750&m=6&q=60&u=t&o=f&l=f&x=1163&y=707","burger.jpg"
)
```

    **************************************************
    burger.jpg
    


```python
apply_model_to_image_raw_bytes(open("burger.jpg", "rb").read())
```

    a white plate with a sandwich and a fork
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_65_1.png)



```python
good_links = [
              "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AABHnbv.img?h=552&w=750&m=6&q=60&u=t&o=f&l=f&x=1163&y=707",
              "https://images.wsj.net/im-139619?width=620&size=1.5",
              "https://www.thoughtco.com/thmb/izS-cAnUqnCdp5IfwxvuTZLG2OY=/768x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-841965988-59c67827685fbe0011219921.jpg",
              "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&w=1000&q=80",
              "https://www.washingtonci.com/skin/frontend/WACI/primary/images/content/products-services/print-signs/traffic-02.jpg",
              "https://media.tacdn.com/media/attractions-splice-spp-674x446/07/25/13/74.jpg",
              "https://awionline.org/sites/default/files/styles/art/public/page/image/dairy%20cow_awa_mike%20suarez%203.jpg",
              "https://cdn.shopify.com/s/files/1/0150/6262/products/the-sill_zz-plant_hover_terracotta_1024x1024.jpg",
              "http://d279m997dpfwgl.cloudfront.net/wp/2017/06/0104_nutonomy02-1000x666.jpg",
              "https://ca-times.brightspotcdn.com/dims4/default/43b9308/2147483647/strip/true/crop/10800x8100+0+0/resize/840x630!/quality/90/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2Ffe%2Ff1%2Fe50f3c604fd8ba870da1edb18f6c%2Fbird1.jpg"
              ]

bad_links = [
             "http://nationalpainreport.com/wp-content/uploads/2014/07/bigstock-Doctor-physician-Isolated-ov-33908342-e1446160270762.jpg",
             "https://www.kimballstock.com/pix/DOG/18/DOG_18_DB0052_01_P.JPG",
             "https://s2.best-wallpaper.net/wallpaper/2560x1600/1904/Cheetah-hunting-deer-speed_2560x1600.jpg",
             "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQoV4-9QvSR5Cz5q8UwIZCMX-76voibk8nfiyRvBhRvlfs8hyQM&s",
             "https://cdn.getyourguide.com/img/tour_img-1667715-146.jpg",
             "https://cdn.britannica.com/42/91642-050-332E5C66/Keukenhof-Gardens-Lisse-Netherlands.jpg",
             "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/10best-cars-group-cropped-1542126037.jpg", 
             "https://cdn.solace.com/wp-content/uploads/2019/01/bg-clouds.jpg",
             "https://img1.nickiswift.com/img/uploads/2017/09/what-the-sun-baby-from-teletubbies-looks-like-today.jpg",
             "https://cdn.images.express.co.uk/img/dynamic/134/590x/A-beach-with-palm-trees-973814.jpg" 
]              
```

Now it's time to find 10 examples where your model works good and 10 examples where it fails! 

You can use images from validation set as follows:
```python
show_valid_example(val_img_fns, example_idx=...)
```

You can use images from the Internet as follows:
```python
! wget ...
apply_model_to_image_raw_bytes(open("...", "rb").read())
```

If you use these functions, the output will be embedded into your notebook and will be visible during peer review!

When you're done, download your noteboook using "File" -> "Download as" -> "Notebook" and prepare that file for peer review!


```python
## Good Job

for i in range(len(good_links)):
    download_utils.download_file(good_links[i],"image_{}.jpg".format(i))
    apply_model_to_image_raw_bytes(open("image_{}.jpg".format(i), "rb").read())
```

    **************************************************
    image_0.jpg
    a white plate with a sandwich and a fork
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_1.png)


    **************************************************
    image_1.jpg
    a man kicking a soccer ball on a field
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_3.png)


    **************************************************
    image_2.jpg
    a view of a river with a mountain range
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_5.png)


    **************************************************
    image_3.jpg
    a group of people standing next to each other
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_7.png)


    **************************************************
    image_4.jpg
    a street sign with a street sign and a tree
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_9.png)


    **************************************************
    image_5.jpg
    a large city with a large clock tower in the background
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_11.png)


    **************************************************
    image_6.jpg
    a cow standing in a field with a white background
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_13.png)


    **************************************************
    image_7.jpg
    a vase with a flower in it and a vase of flowers
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_15.png)


    **************************************************
    image_8.jpg
    a car parked next to a parking meter
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_17.png)


    **************************************************
    image_9.jpg
    a small bird perched on top of a tree branch
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_68_19.png)



```python
## Bad Job

for i in range(len(bad_links)):
    download_utils.download_file(bad_links[i],"image_{}.jpg".format(i))
    apply_model_to_image_raw_bytes(open("image_{}.jpg".format(i), "rb").read())
```

    **************************************************
    image_0.jpg
    a man in a suit and tie holding a white toothbrush
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_1.png)


    **************************************************
    image_1.jpg
    a dog standing on a beach next to a body of water
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_3.png)


    **************************************************
    image_2.jpg
    a group of giraffes standing in a field
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_5.png)


    **************************************************
    image_3.jpg
    a bird is standing on a rock by a fence
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_7.png)


    **************************************************
    image_4.jpg
    a man in a suit and a black and white photo of a man
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_9.png)


    **************************************************
    image_5.jpg
    a fire hydrant on a sidewalk near a tree
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_11.png)


    **************************************************
    image_6.jpg
    a motorcycle parked on the side of a road
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_13.png)


    **************************************************
    image_7.jpg
    a man flying a kite in a field
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_15.png)


    **************************************************
    image_8.jpg
    a child wearing a hat and a hat
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_17.png)


    **************************************************
    image_9.jpg
    a bird flying over a body of water
    


![png](week6_final_project_image_captioning_clean_files/week6_final_project_image_captioning_clean_69_19.png)



```python
'''
### YOUR EXAMPLES HERE ###
image_links = []
for i in range(10):
    image_links.append(input("Paste image link:"))
    download_utils.download_file(image_links[i],"image_{}.jpg".format(i))
    apply_model_to_image_raw_bytes(open("image_{}.jpg".format(i), "rb").read())
'''
```




    '\n### YOUR EXAMPLES HERE ###\nimage_links = []\nfor i in range(10):\n    image_links.append(input("Paste image link:"))\n    download_utils.download_file(image_links[i],"image_{}.jpg".format(i))\n    apply_model_to_image_raw_bytes(open("image_{}.jpg".format(i), "rb").read())\n'



That's it! 

Congratulations, you've trained your image captioning model and now can produce captions for any picture from the  Internet!
