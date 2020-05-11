import tensorflow as tf
import matplotlib.pyplot as plt

class BugModel:

    model = None
    history = None
    modelPath = 'models/bug_model'

    def construct_model(self, vocab_size, embedding_dim, max_length, dropout, embedding_matrix):

        input1 = tf.keras.Input(shape=(max_length,))
        input2 = tf.keras.Input(shape=(max_length,))

        embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix))
        embedding.trainable = False

        embedding1 = embedding(input1)
        embedding2 = embedding(input2)

        bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim))

        vector1 = bi_lstm(embedding1)
        vector1 = tf.keras.layers.Flatten()(vector1)
        vector2 = bi_lstm(embedding2)
        vector2 = tf.keras.layers.Flatten()(vector2)

        x3 = tf.keras.layers.Subtract()([vector1, vector2])
        x3 = tf.keras.layers.Multiply()([x3, x3])

        x1 = tf.keras.layers.Multiply()([vector1, vector1])
        x2 = tf.keras.layers.Multiply()([vector2, vector2])

        x4 = tf.keras.layers.Subtract()([x1, x2])

        x5 = tf.keras.layers.Lambda(self.cosine_distance, output_shape=self.cos_dist_output_shape)([x1, x2])

        x = tf.keras.layers.Concatenate(axis=-1)([x5, x4, x3])

        x = tf.keras.layers.Dense(embedding_dim * 2, activation='relu')(x)

        x = tf.keras.layers.Dropout(rate=dropout)(x)

        pred = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.Model(inputs=[input1, input2], outputs=pred)

        metrics = ['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precission'), tf.keras.metrics.AUC(name='AUC')]
        self.model.compile(loss='hinge',optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),metrics=metrics)
        print(self.model.summary())

    def fit_model(self, training_data, training_labels, testing_data, testing_labels, num_epochs):
        self.history = self.model.fit(training_data, training_labels, epochs=num_epochs, validation_data=(testing_data, testing_labels), verbose=2)

    def predict(self, data):
        return self.model.predict(data)

    def save_model(self):
        self.model.save(self.modelPath)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.modelPath, custom_objects={'cosine_distance': self.cosine_distance, 'cos_dist_output_shape': self.cos_dist_output_shape})
        self.model.summary()

    def plot_graph(self, string):
        plt.plot(self.history.history[string])
        plt.plot(self.history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()

    def plot_graphs(self):
        self.plot_graph('accuracy')
        self.plot_graph('recall')
        self.plot_graph('precission')
        self.plot_graph('AUC')
        self.plot_graph('loss')

    def cosine_distance(self, vests):
        x, y = vests
        x = tf.keras.backend.l2_normalize(x, axis=-1)
        y = tf.keras.backend.l2_normalize(y, axis=-1)
        return -tf.keras.backend.mean(x * y, axis=-1, keepdims=True)

    def cos_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0],1)
    
