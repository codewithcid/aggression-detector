# Aggressiveness-Detection-Deep-Learning

A research project in which a model was trained to detect whether a given audio file contains aggressiveness. This project was extended to also detect whether a given audio file contains an instance of bullying.

## Usage

### Training

The Jupyter notebook can be used to train a model to differentiate between various classes of audio files. To do so, download this project and populate the data folder with pre-sorted .wav audio files. Afterwards, run the notebook to produce .h5, .json, and .yaml files containing the model.

### API Hosting

This project also contains a file ([flask.py](flask.py)) that can be used to host an acoustic AI algorithm. A sample endpoint that uses this code can be found at https://ericthestein.pythonanywhere.com. To invoke this, send a POST request with a body that contains a form, in which the key, "recording" contains a .wav file.

#### API Shortcut

An example usage of the sample endpoint:
![](AggressivenessAPIShortcut.gif)

## Built With

* [TensorFlow](https://www.tensorflow.org/) - a machine learning platform developed by Google
* [Keras](https://keras.io/) - a neural network API
* [Flask](https://palletsprojects.com/p/flask/) - a web application framework


## Author

* **Eric Stein**

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Dr. Anthony Joseph, Pace University - *Oversaw the experimental design and progress of this project*
* Manash Kumar Mandal - *Author of an acoustic deep learning tutorial: https://medium.com/manash-en-blog/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b*
