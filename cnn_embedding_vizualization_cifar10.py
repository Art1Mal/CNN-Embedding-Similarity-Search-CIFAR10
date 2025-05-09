"""
cnn_embedding_visualization_cifar10.py

Deep CNN with global pairwise embedding similarity search on CIFAR-10.
Includes training, evaluation, feature extraction, cosine similarity ranking,
t-SNE visualization, and image pair display.

"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

from tensorflow.keras.layers import (
    Input, Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and normalizes CIFAR-10 dataset."""
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    Y_train = to_categorical(Y_train, 10)
    Y_test = to_categorical(Y_test, 10)
    return X_train, Y_train, X_test, Y_test


def build_model(input_shape: Tuple[int, int, int]) -> Model:
    """Constructs a CNN model using Keras Functional API."""
    inputs = Input(shape=input_shape)

    block1 = Conv2D(99, (3, 3), padding='same', strides=(1, 1),
                    activation='relu', kernel_regularizer=l2(0.001),
                    kernel_initializer='he_normal', use_bias=False)(inputs)
    block1 = BatchNormalization()(block1)

    block2 = Conv2D(99, (3, 3), padding='same', strides=(2, 2),
                    activation='relu', kernel_regularizer=l2(0.001),
                    kernel_initializer='he_normal', use_bias=False)(block1)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(1, 1))(block2)

    block3 = Conv2D(81, (3, 3), padding='same', activation='relu',
                    kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                    use_bias=False)(block2)
    block3 = BatchNormalization()(block3)
    block3 = MaxPooling2D(pool_size=(2, 2))(block3)

    block4 = Conv2D(81, (3, 3), padding='same', activation='relu',
                    kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                    use_bias=False)(block3)
    block4 = BatchNormalization()(block4)
    block4 = MaxPooling2D(pool_size=(2, 2))(block4)

    block5 = Conv2D(81, (3, 3), padding='same', activation='relu',
                    kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                    use_bias=False)(block4)
    block5 = BatchNormalization()(block5)

    block6 = Conv2D(58, (3, 3), padding='same', activation='relu',
                    kernel_regularizer=l2(0.003), kernel_initializer='he_normal',
                    use_bias=False)(block5)
    block6 = BatchNormalization()(block6)
    block6 = Dropout(0.4)(block6)

    block7 = Conv2D(49, (3, 3), padding='same', activation='relu',
                    kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                    use_bias=False)(block6)
    block7 = BatchNormalization()(block7)
    block7 = Dropout(0.3)(block7)

    block8 = Conv2D(49, (3, 3), padding='same', activation='relu',
                    kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                    use_bias=False)(block7)
    block8 = BatchNormalization()(block8)
    block8 = MaxPooling2D(pool_size=(2, 2))(block8)

    block9 = Conv2D(33, (3, 3), padding='same', activation='relu',
                    kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                    use_bias=False)(block8)
    block9 = BatchNormalization()(block9)

    block10 = Conv2D(33, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                     use_bias=False)(block9)
    block10 = BatchNormalization()(block10)

    block11 = Conv2D(18, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                     use_bias=False)(block10)
    block11 = BatchNormalization()(block11)
    block11 = Dropout(0.4)(block11)

    block12 = Conv2D(18, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.003), kernel_initializer='he_normal',
                     use_bias=False)(block11)
    block12 = BatchNormalization()(block12)
    block12 = Dropout(0.3)(block12)

    block13 = Conv2D(15, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                     use_bias=False)(block12)
    block13 = BatchNormalization()(block13)

    block14 = Conv2D(14, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.001), kernel_initializer='he_normal',
                     use_bias=False)(block13)
    block14 = BatchNormalization()(block14)

    x = Flatten()(block14)
    x = Dense(99, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = Dropout(0.4)(x)
    x = Dense(33, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(10, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def main() -> None:
    # Data loading
    X_train, Y_train, X_test, Y_test = load_and_preprocess_data()

    # Model creation
    model = build_model(input_shape=(32, 32, 3))
    model.summary()

    # Compile and train
    model.compile(optimizer=Adadelta(learning_rate=1.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    start = time.time()
    model.fit(X_train, Y_train, epochs=40, batch_size=64, validation_split=0.2)
    print("Training time:", time.time() - start)

    # Evaluate
    loss, acc = model.evaluate(X_test, Y_test, verbose=2)
    print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

    # Metrics
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_true, y_pred, average='macro'))

    # Embedding extraction
    embed_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    embeddings = embed_model.predict(X_test)

    # Similarity analysis
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)

    pairs: List[Tuple[float, Tuple[int, int]]] = []
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            sim = sim_matrix[i, j]
            if len(pairs) < 10:
                heapq.heappush(pairs, (sim, (i, j)))
            else:
                heapq.heappushpop(pairs, (sim, (i, j)))
    closest_pairs = sorted(pairs, key=lambda x: -x[0])

    for sim, (i, j) in closest_pairs:
        print(f"Pair: ({i}, {j}) | Cosine Similarity: {sim:.4f}")

    # t-SNE plot
    X_emb_2D = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_emb_2D[:, 0], X_emb_2D[:, 1], c=y_true, cmap='tab10', s=10)
    plt.title("2D Visualization of Embeddings (t-SNE)")
    plt.colorbar()
    plt.grid(True)
    plt.show()

    # Image pairs
    plt.figure(figsize=(15, 8))
    for idx, (sim, (i, j)) in enumerate(closest_pairs):
        plt.subplot(2, 10, idx + 1)
        plt.imshow(X_test[i])
        plt.axis('off')
        plt.title(f'Img {i}')

        plt.subplot(2, 10, idx + 11)
        plt.imshow(X_test[j])
        plt.axis('off')
        plt.title(f'Close {j}')

    plt.suptitle("Top 10 Most Similar Image Pairs (Global Search)", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()