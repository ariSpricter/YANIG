# Audio Classification

## Requirements
Python 3.11.0 is required.

Install necessary packages:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Training
First, we need to train the model.

Run:
```bash
python train.py
```

This will save a file onto your directory called `classification_model.keras`.

#### Remember
Every time you run this code, you will always save a new classification model onto your directory, if you want to disable this, comment this code in `train.py`
```python
saved_model_path = 'classifier_model.keras'

"""Comment this line to stop saving new models"""
#model.save(saved_model_path, include_optimizer=False)
```

### Step 2: Classification
Run:
```bash
python classify.py
```

This will classify all the data in the `testing_data` and output a confusion matrix.

To test individual classification, uncomment this code in `classify.py`:
```python
# Testing Individual Classification

"""Uncomment this code to test Individual Classification and replace path with testing file path"""
#testing_file_path = "./testing_data/glass_breaking/2-250710-A-39.wav"
#predicted_class, confidence_values = classify_and_plot_file(testing_file_path)
#print(f"Predicted Class: {predicted_class}")
#print(f"Confidence Values: {confidence_values}")

#Testing Directory Classification

"""Make sure to comment this code when testing Individual Classification"""
directory_path = "./testing_data"
classify_and_plot_directory(directory_path=directory_path)
 
```

## License

[MIT](https://choosealicense.com/licenses/mit/)