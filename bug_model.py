import tensorflow as tf
import matplotlib.pyplot as plt

class BugModel:

    model = None
    history = None
    modelPath = 'models/bug_model'

    def constructModel(self, vocab_size, embedding_dim, max_length, label_count):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(label_count, activation='sigmoid')
        ])
        metrics = ['accuracy', tf.keras.metrics.Precision(top_k=10, name='top_10_precission'), tf.keras.metrics.Recall(top_k=10, name='top_10_recall'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy')]
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=metrics)
        print(self.model.summary())

    def fit_model(self, training_data, training_labels, testing_data, testing_labels, num_epochs):
        self.history = self.model.fit(training_data, training_labels, epochs=num_epochs, validation_data=(testing_data, testing_labels), verbose=2)

    def predict(self, data):
        return self.model.predict(data)

    def save_model(self):
        self.model.save(self.modelPath)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.modelPath)
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
        self.plot_graph('top_10_precission')
        self.plot_graph('top_10_recall')
        self.plot_graph('top_10_categorical_accuracy')
        self.plot_graph('loss')
    
