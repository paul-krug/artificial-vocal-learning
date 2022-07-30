
# <a href="https://pypi.org/project/artificial-vocal-learning/"><img alt="ArtificialVocalLearning" src="/ArtificialVocalLearning/logo/ArtificialVocalLearningLogo.svg" height="90"></a>


  
# Quickstart Guide
You can try out this demonstration online on [Google Collab](https://colab.research.google.com/drive/1VhZr3HcW64JOmb2LU5U_LL21_2GzyyBr?usp=sharing) without installing anything locally.

## Installation

    pip install artificial-vocal-learning

## Using the library

Using this framework, vocal learning simulations can be performed easily for any combination of German phonemes.
E.g. the following command will produce each of the vowels /a, e, i, o, u/ once and save the results to 'results/demo/'.

```Python
from ArtificialVocalLearning import demo

demo(
    optimize_units = [ [ 'a' ], [ 'e' ], [ 'i' ], [ 'o' ], [ 'u' ], ],
    runs = [ x for x in range( 0, 1 ) ],
    synthesis_steps = 10,
    out_path = 'results/demo/',
    include_visual_information = False,
    )
```

Thereby, the parameter *runs* is a list that assigns a unique number to each run (repition of the vocal leanring process
for a given phoneme). To generate a phoneme *n* times, *runs* should contain *n* (unique) numbers.
In the example the number of *synthesis_steps* is 10, which ensures that the demonstration is short. To obtain high quality 
speech samples this number should be between 100 and 1000.

Syllables or larger units may be produced like this:

```Python
demo(
    optimize_units = [ [ 'b', 'a' ], ],
    runs = [ x for x in range( 0, 1 ) ],
    synthesis_steps = 1000,
    out_path = 'results/demo/',
    include_visual_information = False,
    trailing_states = 'optimized_preset',
    )
```

This command will produce the syllable /ba/ once, optimized involving 1000 synthesis steps. The parameter *trailing_states* can be "optimize" (which means articulatory parameters will be optimized for all listed phonemes), "vtl_preset" (which means only the first parameter will be optimized but it will be produced in the context of the other listed phonemes generated via VTL preset shapes), or "optimized_preset" (which means only the first parameter will be optimized but it will be produced in the context of the other listed phonemes generated via shapes, which were previously determined by vocal learning and then included in the framework).


Phonemes must be given in SAMPA format. Following phonemes are valid:

['a' 'e' 'i' 'o' 'u' 'E' 'I' 'O' 'U' '2' '9' 'y' 'Y' '@' '6' '?' 'p' 'b' 't' 'd' 'k' 'g' 'f' 'v' 's' 'z' 'S' 'Z' 'C' 'j' 'x' 'R' 'h' 'm' 'n' 'N' 'l']


# Audio Examples
High quality audio examples of German vowels and syllables produced with this framework can be found [here](https://github.com/paul-krug/visual-vocal-learning/tree/master/Stimuli/Manual).

# Description
A python framework for early vocal learning simulations guided by deep-learning based phoneme recognition,
as described in detail in the publication:
``"Artificial Vocal Learning Guided by Phoneme Recognition and Visual Information" by Krug et al.``

