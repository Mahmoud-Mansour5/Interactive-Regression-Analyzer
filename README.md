# CustomRegression-ML-Web

A web-based machine learning application that implements custom linear regression analysis with interactive features and visualizations.

## Features

- Custom implementation of Linear Regression algorithm
- Interactive web interface built with Flask
- CSV data upload and processing
- Real-time predictions
- Data visualization including:
  - Box plots
  - Scatter matrices
  - Prediction error plots
  - Learning curves
- Dynamic feature input handling
- Automatic data preprocessing
- Responsive design

## Tech Stack

- Python 3.x
- Flask (Web Framework)
- Pandas (Data Processing)
- NumPy (Numerical Computations)
- Matplotlib (Data Visualization)
- Scikit-learn (Model Evaluation)
- HTML/CSS (Frontend)

## Project Structure

```
├── app.py                 # Main Flask application
├── custom_regression.py   # Custom Linear Regression implementation
├── data.csv              # Sample dataset
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
└── uploads/             # Directory for uploaded datasets
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CustomRegression-ML-Web.git
cd CustomRegression-ML-Web
```

2. Install required packages:
```bash
pip install flask pandas numpy matplotlib scikit-learn
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Upload Data**: Start by uploading your CSV dataset through the web interface
2. **Train Model**: The system will automatically process your data and train the custom regression model
3. **Visualize**: Explore various visualizations of your data and model performance
4. **Predict**: Use the trained model to make new predictions through the interactive interface

## Features in Detail

### Custom Linear Regression
- Implementation of gradient descent optimization
- Customizable learning rate and iterations
- Automatic feature scaling
- Support for both numerical and categorical features

### Data Preprocessing
- Automatic handling of categorical variables
- Feature scaling
- Missing value detection
- Data validation

### Visualizations
- Interactive plots for data analysis
- Model performance visualization
- Error analysis tools
- Learning curve tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For any queries or suggestions, please open an issue in the GitHub repository. 