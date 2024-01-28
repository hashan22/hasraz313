# hasraz313

I did not use Docker as I don’t have good grip yet on it.

FastAPI Text Classification Service Documentation
Overview
This Python FastAPI application provides a simple API for making text classification predictions using a pre-trained model. The model is based on a neural network trained for text classification and loaded from a saved file (model.h5). The application exposes an endpoint (/predict) that accepts POST requests with input text and returns the predicted label.
Dependencies
•	fastapi
•	uvicorn
•	pandas
•	scikit-learn
•	tensorflow
Usage
1.	Run the FastAPI Application:
Execute the following command in the terminal to start the FastAPI application:
uvicorn model:app –reload

1.	This assumes that your FastAPI application is saved in a file named model.py. The --reload flag enables automatic code reloading during development.
2.	API Endpoint:
The API endpoint for making predictions is available at http://127.0.0.1:8000/predict.
3.	Make Predictions:
Send a POST request to the /predict endpoint with the input text as a form parameter named text. Below is an example using curl:

curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'text=Your input text goes here'

Model and Preprocessing
•	The trained model is loaded from the file model.h5.
•	The label encoder and tokenizer used during training are also loaded from files (label_encoder_classes.pkl and tokenizer_word_index.pkl, respectively).
Notes
•	Ensure that the model file (model.h5) and the required preprocessing files (label_encoder_classes.pkl and tokenizer_word_index.pkl) are present in the same directory as the model.py script.
•	This example assumes a simple text classification model. If your model has specific requirements, adjust the code accordingly.
•	Make sure to have the required dependencies installed in your Python environment before running the FastAPI application:
pip install fastapi uvicorn pandas scikit-learn tensorflow
