# backend/translator/tf.py
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from translator.landmarks import compute_landmark_relationships, extract_landmarks


class PJMClassifier:
    def __init__(self, model_path="models/pjm_model.keras"):
        """
        Initialize the PJMClassifier.

        Args:
            model_path (str): Path to save/load the trained model (relative to project root, using .keras format).
        """
        # Resolve project root as the directory of this script (backend/translator/)
        self.project_root = Path(
            __file__
        ).parent  # Set to the directory of this script: backend/translator/
        self.model_path = self.project_root / model_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes = None

    def load_data(self, data_dir="data/train"):
        """
        Load processed dataset from the specified directory.

        Args:
            data_dir (str): Directory containing processed data (e.g., `data/train/`), relative to project root.

        Returns:
            tuple: (features, labels) as NumPy arrays.
        """
        data_path = self.project_root / data_dir
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory {data_path} does not exist.")

        all_features = []
        all_labels = []

        # Load features and labels for each letter
        for letter_file in data_path.glob("features_*.npy"):
            letter = letter_file.stem.split("_")[1]  # e.g., "A" from "features_A.npy"
            features = np.load(letter_file)
            labels = np.load(data_path / f"labels_{letter}.npy")

            all_features.append(features)
            all_labels.append(labels)

        # Concatenate all data
        X = np.vstack(all_features)  # Shape: (num_samples, 78)
        y = np.hstack(all_labels)  # Shape: (num_samples,)

        # Encode labels (e.g., "A" -> 0, "B" -> 1, etc.)
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = self.label_encoder.classes_

        return X, y_encoded

    def build_model(self, input_shape, num_classes):
        """
        Build an MLP model with batch normalization for PJM gesture classification.

        Args:
            input_shape (int): Number of input features (e.g., 78 for raw coordinates + relationships).
            num_classes (int): Number of output classes (e.g., 22 for letters A-Z).

        Returns:
            tf.keras.Model: Compiled model.
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_shape,)),
                tf.keras.layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.005),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(
                    64,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.005),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Use default Adam learning rate with a scheduler
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(
        self, train_dir="data/train", val_dir="data/val", epochs=50, batch_size=32
    ):
        """
        Train the model using the processed dataset with early stopping and learning rate scheduling.

        Args:
            train_dir (str): Directory with training data (relative to project root).
            val_dir (str): Directory with validation data (relative to project root).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        # Load training and validation data
        X_train, y_train = self.load_data(train_dir)
        X_val, y_val = self.load_data(val_dir)

        # Build the model
        self.model = self.build_model(
            input_shape=X_train.shape[1], num_classes=len(self.classes)
        )

        # Add early stopping (monitor val_accuracy instead of val_loss)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=100, restore_best_weights=True
        )

        # Add learning rate scheduling
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
        )

        # Train the model
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping, lr_scheduler],
        )

        # Save the model in the native Keras format
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """
        Load the trained model if it exists.

        Returns:
            bool: True if the model was loaded, False otherwise.
        """
        if self.model_path.exists():
            self.model = tf.keras.models.load_model(self.model_path)
            # Rebuild label encoder (assumes classes are letters A-Z for simplicity)
            self.label_encoder.fit(
                [
                    chr(i)
                    for i in range(ord("A"), ord("Z") + 1)
                    if chr(i) in "ABCDEFGHIKLMNOPRSTUWYZ"
                ]
            )
            self.classes = self.label_encoder.classes_
            return True
        return False

    def process_frame(self, frame):
        """
        Process a video frame to predict the PJM gesture.

        Args:
            frame (np.ndarray): Video frame in NumPy array format (BGR, from OpenCV).

        Returns:
            tuple: (predicted_letter, detection_result)
                   - predicted_letter (str): Predicted PJM letter (e.g., "A"), or None if no prediction.
                   - detection_result: The raw detection result from MediaPipe Hand Landmarker, or None if no hands detected.

        Raises:
            FileNotFoundError: If the hand landmarker model file is missing.
        """
        # Extract landmarks with error handling
        try:
            detection_result = extract_landmarks(frame)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Cannot perform inference without the hand landmarker model.")
            raise  # Re-raise to signal critical failure
        except Exception as e:
            print(f"Error extracting landmarks from frame: {e}")
            return (
                None,
                None,
            )  # Return None for both predicted letter and detection result

        if not detection_result or not detection_result.hand_landmarks:
            return None, detection_result

        # Compute relationships directly from detection_result
        try:
            relationships = compute_landmark_relationships(
                detection_result
            )  # Shape: (15,)
            # Extract raw coordinates from the first hand for features
            hand_landmarks = detection_result.hand_landmarks[0]  # First detected hand
            raw_coordinates = np.array(
                [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks]
            )  # Shape: (21, 3)
            raw_coordinates = raw_coordinates.flatten()  # Shape: (63,)
            features = np.concatenate([raw_coordinates, relationships])  # Shape: (78,)
        except ValueError as e:
            print(f"Error computing relationships for frame: {e}")
            return (
                None,
                detection_result,
            )  # Return None for predicted letter, but still return detection_result

        # Load model if not already loaded
        if self.model is None:
            if not self.load_model():
                raise RuntimeError(
                    "No trained model available. Please train the model first."
                )

        # Predict
        features = features.reshape(1, -1)  # Shape: (1, 78)
        prediction = self.model.predict(features, verbose=0)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_letter = self.label_encoder.inverse_transform([predicted_class_idx])[
            0
        ]

        return predicted_letter, detection_result

    def evaluate(self, test_dir="data/test"):
        """
        Evaluate the model on the test dataset.

        Args:
            test_dir (str): Directory with test data (relative to project root).

        Returns:
            tuple: (loss, accuracy) on the test set.
        """
        # Load test data
        X_test, y_test = self.load_data(test_dir)

        # Load model if not already loaded
        if self.model is None:
            if not self.load_model():
                raise RuntimeError(
                    "No trained model available. Please train the model first."
                )

        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
