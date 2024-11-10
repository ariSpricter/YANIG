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

To save the model, pass the `--save` argument:
```bash
python train.py --save
```
This will save the model into a file named `classification_model.keras`. This file is needed for classifying audio.


### Step 2: Classification
Run:
```bash
python classify.py
```

This will classify all data in the `testing_data` and output a confusion matrix.

To do individual classification, pass the `--file` argument with a file path to the specified audio:
```bash
python classify.py --file path/to/your_audio.wav
# Example:
python classify.py --file testing_data/glass_breaking/2-106881-A-39.wav
```
This will output the model's prediction alongside its' confidence values.

## License

[MIT](https://choosealicense.com/licenses/mit/)