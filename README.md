# Ligand_binding_keras

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam

    # Load the dataset of ligand-protein complexes and their binding free energies
    data = pd.read_csv('dataset.csv')

    # Extract the features and target variable
    X = data.drop(['Binding Free Energy'], axis=1).values
    y = data['Binding Free Energy'].values.reshape(-1, 1)

    # Scale the features using standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define a deep neural network model using Keras
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    # Compile the model and set the optimizer and loss function
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # Train the model on the training data
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the testing data
    mse = model.evaluate(X_test, y_test)
    print('Mean squared error:', mse)

    # Use the model to predict the binding free energy of a new ligand-protein complex
    new_data = np.array([[-0.5, 0.1, -0.3, 0.4, -0.5, 0.6]])  # Replace with your own data
    new_data = scaler.transform(new_data)
    predicted_energy = model.predict(new_data)
    print('Predicted binding free energy:', predicted_energy[0][0])
    
    
  
  In this code, `dataset.csv` is a CSV file containing a dataset of ligand-protein complexes and their binding free energies. The dataset should include features of the ligand and protein, such as their electrostatic properties, shape complementarity, and solvation effects. The target variable, binding free energy, should also be included.

The code first loads the dataset and extracts the features and target variable using the drop and values methods of pandas. It then scales the features using standardization to improve the performance of the deep neural network.

The data is split into training and testing sets using the **train_test_split** function from scikit-learn. The deep neural network model is defined using the Keras Sequential API, which allows you to easily stack layers of the neural network. The model architecture includes three dense layers, with the first two using the rectified linear unit (ReLU) activation function and dropout regularization to prevent overfitting.

The model is compiled using the Adam optimizer and mean squared error (MSE) loss function. It is then trained on the training data using the fit method, with a batch size of 32 and 50 epochs.

The model is evaluated on the testing data using the evaluate method, which calculates the mean squared error (MSE) between the predicted and actual binding free energies.
