from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
from custom_regression import CustomLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import uuid
import json
import numpy as np
IMAGE_FOLDER = 'static/images'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Car data will be loaded from the uploaded dataset
CAR_MAKES = []
CAR_MODELS = {}
CYLINDERS = []
MODEL_YEARS = []

# Global variables to store the model and feature names
current_model = None
feature_names = None
column_values = None
model_trained = False

def save_plot(fig):
    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(IMAGE_FOLDER, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return f"/static/images/{filename}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_form():
    return render_template('upload.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global current_model, feature_names, column_values, model_trained
    
    if not model_trained:
        return "Please train a model first by uploading a dataset"
    
    if request.method == 'POST':
        try:
            # Get input values from form
            input_data = {}
            for feature in feature_names:
                value = request.form.get(feature)
                if value is None or value == '':
                    return f"Missing value for {feature}"
                
                # Convert value to appropriate type based on original data
                try:
                    # Try to convert to float first
                    input_data[feature] = float(value)
                except ValueError:
                    # If conversion fails, keep as string
                    input_data[feature] = value
            
            # Create a DataFrame with the original features
            original_input = pd.DataFrame([input_data])
            
            # Apply the same preprocessing as during training
            # One-hot encode categorical variables
            processed_input = pd.get_dummies(original_input)
            
            # Ensure all columns from training are present
            for col in current_model['feature_names_in_']:
                if col not in processed_input.columns:
                    processed_input[col] = 0
            
            # Reorder columns to match training data
            processed_input = processed_input[current_model['feature_names_in_']]
            
            # Scale the features using the same parameters as training
            processed_input_scaled = (processed_input - current_model['X_mean']) / current_model['X_std']
            
            # Make prediction
            prediction = current_model['model'].predict(processed_input_scaled.values)[0]
            
            return render_template('prediction_result.html', 
                                prediction=prediction,
                                input_data=input_data,
                                feature_names=feature_names)
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    
    # Get unique values for each feature from the dataset
    feature_options = {}
    for feature in feature_names:
        if feature in column_values:
            feature_options[feature] = column_values[feature]
    
    return render_template('predict.html', 
                         feature_names=feature_names,
                         feature_options=feature_options)

def create_box_plot(X, y):
    # Create box plots for first 4 features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(X.columns[:4]):
        axes[idx].boxplot(X[feature])
        axes[idx].set_title(f'Box Plot of {feature}')
        axes[idx].set_ylabel('Value')
    
    plt.tight_layout()
    return fig

def create_scatter_matrix(X, y):
    # Create scatter matrix for first 4 features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(X.columns[:4]):
        axes[idx].scatter(X[feature], y, alpha=0.6)
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Target')
        axes[idx].set_title(f'{feature} vs Target')
    
    plt.tight_layout()
    return fig

def create_prediction_error_plot(y_test, predictions):
    # Create prediction error plot
    fig, ax = plt.subplots(figsize=(10, 6))
    error = y_test - predictions
    ax.scatter(predictions, error, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error Plot')
    return fig

def create_learning_curve_plot(X, y):
    # Create simple learning curve by varying training set size
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_scores = []
    test_scores = []
    
    for size in train_sizes:
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-size)
            
            # Scale the features
            X_mean = X_train.mean()
            X_std = X_train.std().replace(0, 1)
            X_train_scaled = (X_train - X_mean) / X_std
            X_test_scaled = (X_test - X_mean) / X_std

            # Convert to numpy arrays and ensure float64 type
            X_train_scaled = X_train_scaled.values.astype('float64')
            X_test_scaled = X_test_scaled.values.astype('float64')
            y_train = y_train.values.astype('float64')
            y_test = y_test.values.astype('float64')

            # Train model
            model = CustomLinearRegression(learning_rate=0.01, n_iterations=1000)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate MSE, handling potential NaN values
            train_mse = np.mean((y_train - train_pred) ** 2)
            test_mse = np.mean((y_test - test_pred) ** 2)
            
            # Only append scores if they are valid numbers
            if not (np.isnan(train_mse) or np.isnan(test_mse) or 
                   np.isinf(train_mse) or np.isinf(test_mse)):
                train_scores.append(train_mse)
                test_scores.append(test_mse)
            
        except Exception as e:
            print(f"Error at training size {size}: {str(e)}")
            continue
    
    # Create plot only if we have valid scores
    if len(train_scores) > 0 and len(test_scores) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes[:len(train_scores)], train_scores, 'o-', label='Training Error')
        ax.plot(train_sizes[:len(test_scores)], test_scores, 'o-', label='Testing Error')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Learning Curve')
        ax.legend()
        return fig
    else:
        # Create an empty plot with a message if no valid scores
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Could not generate learning curve due to numerical instability',
                horizontalalignment='center', verticalalignment='center')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Learning Curve')
        return fig

def get_first_five_columns(X):
    # Get the first 5 column names
    return X.columns[:5].tolist()

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_model, feature_names, column_values, model_trained
    
    if 'file' not in request.files:
        return render_template('upload.html', error="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error="No selected file")
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load CSV
            df = pd.read_csv(filepath)

            # Capture original shape and info
            original_shape = df.shape
            buf_before = io.StringIO()
            df.info(buf=buf_before)
            original_info = buf_before.getvalue()

            # Clean data
            df.drop_duplicates(inplace=True)
            df_cleaned = df.copy()

            # Remove rows where target variable (MSRP) is NaN
            df_cleaned = df_cleaned.dropna(subset=['MSRP'])

            # Convert MSRP to numeric, removing any currency symbols or commas
            df_cleaned['MSRP'] = pd.to_numeric(df_cleaned['MSRP'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce')

            # Separate features and target
            X = df_cleaned.iloc[:, :-1]  # all columns except last
            y = df_cleaned.iloc[:, -1]   # last column is target (MSRP)

            # Get first 5 columns and their unique values
            feature_names = get_first_five_columns(X)
            column_values = {}
            for col in X.columns:
                # Get unique values
                unique_values = X[col].unique()
                # Sort numeric values numerically, strings alphabetically
                if pd.api.types.is_numeric_dtype(X[col]):
                    unique_values = sorted([x for x in unique_values if pd.notna(x)])
                else:
                    unique_values = sorted([x for x in unique_values if pd.notna(x)], key=str)
                # Convert all values to strings for display
                column_values[col] = [str(val) for val in unique_values]

            # Store original feature names before encoding
            original_features = X.columns.tolist()

            # Identify numeric and categorical columns
            numeric_features = []
            categorical_features = []
            
            for col in X.columns:
                # Try to convert to numeric
                try:
                    X[col] = pd.to_numeric(X[col], errors='raise')
                    numeric_features.append(col)
                except (ValueError, TypeError):
                    categorical_features.append(col)

            # Handle missing values in numeric columns
            for col in numeric_features:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)

            # Handle missing values and encode categorical columns
            for col in categorical_features:
                # Fill missing values with mode
                mode_val = X[col].mode()[0]
                X[col] = X[col].fillna(mode_val)
                # Convert to category type
                X[col] = X[col].astype('category')

            # One-hot encode categorical variables
            X = pd.get_dummies(X, columns=categorical_features)

            # Remove any remaining NaN values
            X = X.fillna(0)

            # Convert all columns to float64
            X = X.astype('float64')

            # Capture cleaned shape and info
            cleaned_shape = X.shape
            buf_after = io.StringIO()
            X.info(buf=buf_after)
            cleaned_info = buf_after.getvalue()

            # Create data exploration plots
            numeric_features_for_plots = X.select_dtypes(include=['float64', 'int64']).columns[:4]
            box_plot_url = save_plot(create_box_plot(X[numeric_features_for_plots], y))
            scatter_matrix_url = save_plot(create_scatter_matrix(X[numeric_features_for_plots], y))

            # Train/test split and model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Scale the features
            X_mean = X_train.mean()
            X_std = X_train.std().replace(0, 1)  # Replace zero std with 1 to avoid division by zero
            X_train_scaled = (X_train - X_mean) / X_std
            X_test_scaled = (X_test - X_mean) / X_std

            # Convert to numpy arrays and ensure float64 type
            X_train_scaled = X_train_scaled.values.astype('float64')
            X_test_scaled = X_test_scaled.values.astype('float64')
            y_train = y_train.values.astype('float64')
            y_test = y_test.values.astype('float64')

            # Train the model
            model = CustomLinearRegression(learning_rate=0.01, n_iterations=1000)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, predictions)

            # Store model and scaling parameters globally
            current_model = {
                'model': model,
                'feature_names_in_': X.columns,
                'X_mean': X_mean,
                'X_std': X_std
            }
            model_trained = True

            # Create model evaluation plots
            error_plot_url = save_plot(create_prediction_error_plot(y_test, predictions))
            learning_curve_url = save_plot(create_learning_curve_plot(X, y))

            # Original plots
            numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                first_feature = numeric_cols[0]
                fig1, ax1 = plt.subplots()
                ax1.scatter(X[first_feature], y, alpha=0.6)
                ax1.set_title(f'{first_feature} vs Target (Before Training)')
                ax1.set_xlabel(first_feature)
                ax1.set_ylabel('Target')
                plot1_url = save_plot(fig1)
            else:
                plot1_url = None

            fig2, ax2 = plt.subplots()
            ax2.scatter(y_test, predictions, alpha=0.6, color='green')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ideal line
            ax2.set_title('Actual vs Predicted (After Training)')
            ax2.set_xlabel('Actual')
            ax2.set_ylabel('Predicted')
            plot2_url = save_plot(fig2)

            return render_template('training_results.html',
                                original_shape=original_shape,
                                original_info=original_info,
                                cleaned_shape=cleaned_shape,
                                cleaned_info=cleaned_info,
                                mse=mse,
                                feature_names=feature_names,
                                box_plot_url=box_plot_url,
                                scatter_matrix_url=scatter_matrix_url,
                                error_plot_url=error_plot_url,
                                learning_curve_url=learning_curve_url,
                                plot1_url=plot1_url,
                                plot2_url=plot2_url)

        except Exception as e:
            import traceback
            print(traceback.format_exc())  # Print the full error traceback
            return render_template('upload.html', error=f"Error processing file: {str(e)}")
    
    return render_template('upload.html', error="Invalid file type. Please upload a CSV file.")

if __name__ == '__main__':
    app.run(debug=True)
