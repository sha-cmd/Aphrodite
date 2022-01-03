
import tensorflow as tf
import joblib

from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.models import load_model

global vectorize_layer
# Vectorize the data.
vectorize_layer = TextVectorization(
    max_tokens=100000,
    output_mode="int",
    output_sequence_length=256,
)


def cost_metric(y_true, y_pred):
    unit_cost_of_a_bad_buzz = 1

    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

    cost = K.log(tn / (tn + fp)) * unit_cost_of_a_bad_buzz
    if tf.math.is_nan(cost) or tf.math.is_inf(cost):
        cost = unit_cost_of_a_bad_buzz * 10.0

    return cost


def vectorize_text(text, label):
    global vectorize_layer
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def create(text):
    global vectorize_layer
    dir_list = ['data/train', 'data/test']
    batch_size = 256
    dataset_train = keras.preprocessing.text_dataset_from_directory(
        dir_list[0], batch_size=batch_size, seed=42, subset='training', validation_split=0.2)


    text_ds = dataset_train.map(lambda x, y: x)

    train_ds = dataset_train.map(vectorize_text)
    # Let's call `adapt`:
    vectorize_layer.adapt(text_ds)
    action = 'nature'
    recall = 0.5
    specificity = 0.5
    sensitivity = 0.5
    optimizer = 'adam'
    model = load_model('models/lstm_model_' + action + '.h5', custom_objects={'cost_metric': cost_metric})

    # A string input
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    # Turn strings into vocab indices
    indices = vectorize_layer(inputs)
    # Turn vocab indices into predictions
    outputs = model(indices)

    # Our end to end model
    end_to_end_model = tf.keras.Model(inputs, outputs)

    end_to_end_model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=['acc', 'AUC',
                                                                  tf.keras.metrics.Precision(),
                                                                  tf.keras.metrics.Recall(),
                                                                  tf.keras.metrics.PrecisionAtRecall(recall),
                                                                  tf.keras.metrics.SensitivityAtSpecificity(
                                                                      specificity),
                                                                  tf.keras.metrics.SpecificityAtSensitivity(
                                                                      sensitivity),
                                                                  cost_metric
                                                                  ]
    )

    raw_text_data = tf.convert_to_tensor([
        [text],
    ])

    end_to_end_model.save_weights('models/end_to_end')
    predictions = end_to_end_model(raw_text_data)

    print('Prediction : ', 'positif' if predictions.numpy()[0][0] > 0.5 else 'negatif')

    return 'Prediction : ', 'positif' if predictions.numpy()[0][0] > 0.5 else 'negatif', predictions.numpy()[0][0]

def predict(text):


    model = load_model('models/end_to_end', custom_objects={'cost_metric': cost_metric})

    raw_text_data = tf.convert_to_tensor([
        [text],
    ])
    predictions = model(raw_text_data)


if __name__ == "__main__":
    create('I enjoy playing the piano')
    predict('I enjoy playing the piano')