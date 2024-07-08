
# MACHINE LEARNING OEL PROJECT

## Created by Muhammad Adan and Muhib Ullah

## INSURANCE PRICE PREDICTOR
This project predicts insurance prices based on various factors such as Age, BMI, Smoker status, Number of Children, and Region using machine learning models.
```
### Project Structure

backend/
├── main.py
├── model.pkl
├── templates/
│   ├── image.jpg
│   └── index.html
└── ml_models/
    ├── dataset/
    │   └── post_processed_dataset.csv
    ├── json/
    │   └── evaluation_matrix.json
    └── regression_model/
        ├── linear_regression_model.py
        ├── python_regression_model.py
        ├── random_forest_model.py
        └── trainModel.py
```

### Main Libraries Used

- Python
- Flask
- Pandas
- Joblib
- Numpy

### How to Run the Project

1. Navigate to the `backend` directory:

    ```sh
    cd backend
    ```

2. Run the `main.py` file:

    ```sh
    python main.py
    ```

3. A local server will be created, running the project. Open your browser and navigate to `http://127.0.0.1:5000` to view the application.

## Exploratory Data Analysis (EDA)

For Exploratory Data Analysis follow the given link: [EDA](https://colab.research.google.com/drive/1kfFCQZcIl4oTXN4IXBXPMCblLY90QNqd?usp=sharing).

Enjoy predicting insurance prices with our machine learning models!
