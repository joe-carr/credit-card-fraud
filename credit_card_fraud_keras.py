import tensorflow as tf
import matplotlib.pyplot as plt
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow_addons as tfa

def data():
    credit_data = pd.read_csv("creditcard.csv", delimiter=",")
    features_df = credit_data.copy()
    features_df = shuffle(features_df, random_state=1)
    y = np.array(features_df.pop("Class"))
    x = np.array(features_df)
    pos = np.sum(y)
    neg = np.size(y) - pos
    w_neg = (1.0 / neg) * np.size(y) / 2
    w_pos = (1.0 / pos) * np.size(y) / 2
    class_weight = {0: w_neg, 1: w_pos}
    val_no_samples = int(len(y) * 0.2)
    train_x = x[:-val_no_samples]
    val_x = x[-val_no_samples:]
    train_y = y[:-val_no_samples]
    val_y = y[-val_no_samples:]
    mean = train_x.mean(axis=0)
    train_x -= mean
    std = train_x.std(axis=0)
    train_x /= std
    val_x -= mean
    val_x /= std
    return train_x, train_y, val_x, val_y, class_weight


def create_model(train_x, train_y, val_x, val_y, class_weight):
    input_tensor = tf.keras.Input(shape=(30,))
    dense_1 = tf.keras.layers.Dense({{choice([64, 128, 256, 512, 1024])}},activation='relu')(input_tensor)
    dropout_1 = tf.keras.layers.Dropout({{uniform(0, 0.6)}})(dense_1)
    dense_2 = tf.keras.layers.Dense({{choice([64, 128, 256, 512, 1024])}},activation='relu')(dropout_1)
    dropout_2 = tf.keras.layers.Dropout({{uniform(0, 0.6)}})(dense_2)
    dense_3 = tf.keras.layers.Dense({{choice([64, 128, 256, 512, 1024])}},activation='relu')(dropout_2)
    dropout_3 = tf.keras.layers.Dropout({{uniform(0, 0.6)}})(dense_3)
    output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_3)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor, name="credit_card_fraud_detection")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tfa.metrics.F1Score(num_classes=1, threshold=0.5, name='f1_score'), "accuracy"])
    history = model.fit(train_x, train_y, epochs=50, batch_size=2048, validation_data=(val_x, val_y), verbose=1,
                                  class_weight=class_weight)
    val_loss = np.amin(history.history['val_loss'])
    val_f1_score = np.amax(history.history['val_f1_score'])
    return {'loss': val_loss, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          eval_space=True,
                                          max_evals=100,
                                          trials=Trials())


print("\n Best performing model chosen hyper-parameters:")
print(best_run)
best_model.save('best_hyperas_model.h5')
train_x, train_y, val_x, val_y, class_weight = data()

metrics = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tfa.metrics.F1Score(num_classes=1, threshold=0.5, name='f1_score')
]

best_model = tf.keras.models.load_model('best_hyperas_model.h5')
config = best_model.get_config()
best_model = tf.keras.Model.from_config(config)
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

callbacks = [tf.keras.callbacks.ModelCheckpoint("best_f1_score_at_epoch_{epoch}.h5", monitor='val_f1_score', mode='max', save_best_only=True)]

history = best_model.fit(train_x, train_y, epochs=50, batch_size=2048, validation_data=(val_x, val_y), verbose=1,
                                  class_weight=class_weight, callbacks = callbacks)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
false_pos = history.history['fp']
val_false_pos = history.history['val_fp']
false_neg = history.history['fn']
val_false_neg = history.history['val_fn']
precision = history.history['precision']
val_precision = history.history['val_precision']
recall = history.history['recall']
val_recall = history.history['val_recall']
AUC = history.history['auc']
val_AUC = history.history['val_auc']
f1_score = history.history['f1_score']
val_f1_score = history.history['val_f1_score']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, recall, 'b', label='Training recall')
plt.plot(epochs, val_recall, 'r', label='Validation recall')
plt.title('Training and validation recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.show()

plt.plot(epochs, precision, 'b', label='Training precision')
plt.plot(epochs, val_precision, 'r', label='Validation precision')
plt.title('Training and validation precision')
plt.xlabel('Epochs')
plt.ylabel('precision')
plt.legend()
plt.show()

plt.plot(epochs, AUC, 'b', label='Training AUC')
plt.plot(epochs, val_AUC, 'r', label='Validation AUC')
plt.title('Training and validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()

plt.plot(epochs, f1_score, 'b', label='Training F1 score')
plt.plot(epochs, val_f1_score, 'r', label='Validation F1 score')
plt.title('Training and validation F1 score')
plt.xlabel('Epochs')
plt.ylabel('F1 score')
plt.legend()
plt.show()

best_f1_score = np.amax(val_f1_score)
best_f1_score_index = val_f1_score.index(best_f1_score)
best_fp = history.history['val_fp'][best_f1_score_index]
best_fn = history.history['val_fn'][best_f1_score_index]

print("Final validation F1 score: ", best_f1_score)
print("Final number of validation False Positives: ", best_fp)
print("Final number of validation False Negatives: ", best_fn)