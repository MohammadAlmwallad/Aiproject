from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import warnings
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP: known incorrect sRGB profile.*")


# Load Data
data = tf.keras.utils.image_dataset_from_directory('data', image_size=(250, 250), batch_size=64)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Display the first four images and their labels
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].set_title(str(batch[1][idx]))

plt.show()

#Scale Data
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

#Split Data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1) + 1
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)



data = data.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
data = data.map(lambda x, y: (tf.image.random_flip_up_down(x), y))
data = data.map(lambda x, y: (tf.image.random_brightness(x, max_delta=0.1), y))
data = data.map(lambda x, y: (tf.image.random_contrast(x, lower=0.9, upper=1.1), y))


#Build Deep Learning Model

model = Sequential()
model.add(Conv2D(32, (3, 3), 1, activation='relu', input_shape=(250, 250, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#Train
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3 * 0.9 ** epoch
)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback, lr_scheduler])

#Plot Performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Evaluate
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
y_true = []
y_pred = []

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

    y_true.extend(y)
    y_pred.extend((yhat > 0.5).astype(int))

print("Precision:", pre.result().numpy())
print("Recall:", re.result().numpy())
print("Accuracy:", acc.result().numpy())

# F1 Score
f1 = f1_score(y_true, y_pred)

print("F1 Score:", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model.save(os.path.join('models', 'imageclassifier.h5'))