# FFNN + Handwriting recognition

Custom library for implementing feedforward neural networks using back propagation.
This library is then used for handwriting recognition with a gui (MNIST database).
I recreated the algorithm from scratch, without using any external library, to fully understand the process.

## Usage

### Create a model

Create, train and save a model inside `output/model.pkl`.

```bash
python3 create_model.py
```

To modify the model, edit the variable `netword` inside `create_model.py`

### Test the model

Launch a gui for manually testing the model

```bash
python3 main.py
```

## Requirements

* tkinter : `sudo apt install python3-tk`

## Credits

User [nikhilkumarsingh](https://github.com/nikhilkumarsingh) for the gui [base code](https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06).