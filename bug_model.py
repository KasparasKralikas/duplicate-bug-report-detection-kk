import tensorflow as tf
import matplotlib.pyplot as plt

class BugModel:

    model = None
    history = None
    modelPath = 'models/bug_model'

    def constructModel(self, vocab_size, embedding_dim, max_length):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
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
        self.plot_graph("accuracy")
        self.plot_graph("loss")
    
