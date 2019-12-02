![](img/banner.svg)
# Handwriting Synthesis
Implementation of the handwriting synthesis experiments in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves.  The implementation closely follows the original paper, with a few slight deviations, and the generated samples are of similar quality to those presented in the paper.

## Installation

The project can be installed into the current Python environment with
```bash
pip install .
```

#### PNG Image Export

For generated images to be exported as PNG files, a few additional steps are required via the CairoSVG dependency. On OS X, the `cairo` and `libffi` system packages must be installed (via `brew`), and the following lines should be added to the `.bash_profile` ([source](https://github.com/Kozea/WeasyPrint/issues/737#issuecomment-439112117)): 

```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8 
```

Note that these steps have only been tested in an OS X environment, and may differ on other platforms.

## Usage
```python
from handwriting_synthesis.hand import Hand

lines = [
    "Now this is a story all about how",
    "My life got flipped turned upside down",
    "And I'd like to take a minute, just sit right there",
    "I'll tell you how I became the prince of a town called Bel-Air",
]
biases = [.75 for i in lines]
styles = [9 for i in lines]
stroke_colors = ['red', 'green', 'black', 'blue']
stroke_widths = [1, 2, 1, 2]

hand = Hand()
hand.write(
    filename='img/usage_demo.svg',
    lines=lines,
    biases=biases,
    styles=styles,
    stroke_colors=stroke_colors,
    stroke_widths=stroke_widths
)
```
![](img/usage_demo.svg)

A pretrained model is included, but if you'd like to train your own, read <a href='https://github.com/sjvasquez/handwriting-synthesis/tree/master/data/raw'>these instructions</a>.

## Demonstrations
Below are a few hundred samples from the model, including some samples demonstrating the effect of priming and biasing the model.  Loosely speaking, biasing controls the neatness of the samples and priming controls the style of the samples. The code for these demonstrations can be found in `demo.py`.

### Demo #1:
The following samples were generated with a fixed style and fixed bias.

**Smash Mouth – All Star (<a href="https://www.azlyrics.com/lyrics/smashmouth/allstar.html">lyrics</a>)**
![](img/all_star.svg)

### Demo #2
The following samples were generated with varying style and fixed bias.  Each verse is generated in a different style.

**Vanessa Carlton – A Thousand Miles (<a href="https://www.azlyrics.com/lyrics/vanessacarlton/athousandmiles.html">lyrics</a>)**
![](img/downtown.svg)

### Demo #3
The following samples were generated with a fixed style and varying bias.  Each verse has a lower bias than the previous, with the last verse being unbiased.

**Leonard Cohen – Hallelujah (<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">lyrics</a>)**
![](img/give_up.svg)

## Contribute
Areas for future contributions:
- General project documentation and commenting
- Structural cleanup and optimization
- Updates to newer TensorFlow versions and deprecation updates
- Completely packaging this project for PyPi
