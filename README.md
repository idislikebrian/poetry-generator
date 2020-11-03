# poetry-generator

[![GitHub license](https://img.shields.io/github/license/PraveenKumarSridhar/poetry-generator?style=for-the-badge)](https://github.com/PraveenKumarSridhar/poetry-generator/blob/master/LICENSE)

Mine a poets poetry from [poets.org](https://poets.org) then use that collection of poetry to train a model using TensorFlow to write short poems.

Check out the medium post that I followed to set up the TensorFlow part [here](https://medium.com/@prasri.pk/can-we-write-a-sonnet-like-its-the-middle-ages-f3c06ecb690).


## DATA:
<hr/>

The data used is scraped with [BeautifulSoup 4.9.0](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) from poets.org. A sample of this data for example, from Pablo Neruda would look like:

```
Today is dead winter in the forgotten land
that comes to visit me, with a cross on the map
and a volcano in the snow, to return to me,
to return again the water
fallen on the roof of my childhood.
Today when the sun began with its shafts
to tell the story, so clear, so old,
the slanting rain fell like a sword,
the rain my hard heart welcomes.
```

## Environment and tools Used:
<hr/>

```
1. Jupyter Notebook
2. Numpy
3. Matplotlib
4. Tensorflow
```
Plus...

```
5. BeautifulSoup
```

## Required package installation:
<hr/>

```
pip install tensorflow==2.1.0
pip install numpy
```
As well as..

```
pip install beautifulsoup4
```

## Results (via PraveenKumarSridhar)
<hr/>

### Character level 

Training accuracy vs epochs 

<img align="center" alt="Training accuracy vs epochsr"  src="https://raw.githubusercontent.com/PraveenKumarSridhar/poetry-generator/develop/src/Sonnets/Plots/accuracy_plot.png" />

<br/>

Training loss vs epochs 

<img align="center" alt="Training loss vs epochs "  src="https://raw.githubusercontent.com/PraveenKumarSridhar/poetry-generator/develop/src/Sonnets/Plots/loss_plot.png" />
