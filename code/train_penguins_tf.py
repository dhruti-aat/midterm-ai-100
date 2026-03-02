import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow import keras
from tensorflow.keras import layers

DATA_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"

def main():
    df = pd.read_csv(DATA_URL)

    feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    target_col = "species"

    df = df[[target_col] + feature_cols].dropna()

    class_names = sorted(df[target_col].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(class_names)}

    X = df[feature_cols].astype(float).values
    y = df[target_col].map(class_to_id).astype(int).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(len(class_names), activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=[es],
        verbose=1
    )

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print("\nClass names:", class_names)
    print("Test accuracy:", round(acc, 4))
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=class_names))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    model.save("penguins_mlp.keras")
    print("\nSaved model to penguins_mlp.keras")

if __name__ == "__main__":
    main()
