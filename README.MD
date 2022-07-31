# Tensorflow MNIST Model Api with Flask

Using Flask to build a Restful API Server that will return the result of the ML model MNSIT.

Integration with Flask, request, jsonify tensorflow, tensorflow-keras data-set and scipy misc extensions.

### Extension:
- Flask jsonify: [Flask.jsonify](https://tedboy.github.io/flask/generated/flask.jsonify.html)

- TensorFlow: [Tensorflow 2](https://www.tensorflow.org/api_docs/python/tf)

- Numpy: [NumPy](https://numpy.org/)

- Scipy Misc: [scipy.misc](https://docs.scipy.org/doc/scipy/reference/misc.html)



## Installation

Install with pip:

```
$ pip install -r requirements.txt
```




 
## Run Flask
### Run flask for develop
```
$ python app.py
```
In flask, Default port is `5000`

Swagger document page:  `http://127.0.0.1:5000/api`

### Run flask for production

** Run with gunicorn **

In  webapp/

```
$ gunicorn -w 4 -b 127.0.0.1:5000 run:app

```

* -w : number of worker
* -b : Socket to bind



