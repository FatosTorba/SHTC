### A Study on Hierarchical Text Classification as a Seq2seq Task
This is the official implementation for the paper titled "A Study on Hierarchical Text Classification as a Seq2seq Task."
![Alt text](./SHTC.svg " SHTC illustration. The Target generator offers 4 target sequence options, green nodes denote correct
document labels. The constrained module is optional, without it, SHTC operates like a standard T5 architecture. During auto-regressive decoding, this latter computes the probability distribution over the next
possible tokens set based on prior predictions and global hierarchy, while nullifying others (constrained
probs). To enhance clarity, we only consider a BFS target. For instance, if the model has already generated
“A -” the constrained module will distribute the probabilities over {B, C} (indicated by the green arrow
and green box). The purple and yellow arrows illustrate the next two constrained generation steps.")
### Abstract
With the progress of generative neural models, Hierarchical Text Classification
(HTC) can be cast as a generative task. In this case, given an input text, the model generates
the sequence of predicted class labels taken from a label tree of arbitrary width and depth.
Treating HTC as a generative task introduces multiple modeling choices. These choices vary
from choosing the order for visiting the class tree and therefore defining the order of generat-
ing tokens, choosing either to constrain the decoding to labels that respect the previous level
predictions, up to choosing the pre-trained Language Model itself. Each HTC model therefore
differs from the others from an architectural standpoint, but also from the modeling choices
that were made. Prior contributions lack transparent modeling choices and open implemen-
tations, hindering the assessment of whether model performance stems from architectural or
modeling decisions. For these reasons, we propose with this paper an analysis of the impact
of different modeling choices along with common model errors and successes for this task.
This analysis is based on an open framework coming along this paper that can facilitate the
development of future contributions in the field by providing datasets, metrics, error analysis
toolkit and the capability to readily test various modeling choices for one given model.

### Project components

The project consists of following parts:
- **`constrained_module`** : Contains code for constrained sequence generation for both training and inference with beam search. 
- **`dataset`**:  Repository for the three datasets used in this paper:  reuters corpus [RCV1V2](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm) , Blurb Genre Collection [BGC](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html) and Web-Of-Science [WOS](https://data.mendeley.com/datasets/9rw3vkcfy4/2). Due to legal issues, RCV1V2 dataset is not available in this repository. Researchers must sign an agreement to obtain this dataset.  
- **`metrics`**: 
  - F1 scores: F1-micro , F1-macro, per level and global.
  - Constrained-F1 scores: C-F1 micro, C-F1 macro, per level and global.
  - Hierarchical inconsistencies : Hierarchical Consistency Rate (HCR), True Positive Hierarchical.
Consistency Rate (TP-HCR) and False positive Hierarchical Consistency Rate (FP-HCR).
  - Depth of Prediction Rate.
- **`model`** : T5 model used for hierarchical text classification.
- **`target_generator`**: Four options for the target sequence: Depth-First Search (DFS), Breadth-First Search (BFS), Bottom-Top BFS,  Bottom-Top pseudo-DFS.

### Requirements
The main dependencies to run this code are:
- Python  3.7.9
- PyTorch  1.7.0
- Transformers  2.9.0

### Train and test

To train and test the model, one can simply execute the following command:

```python main.py --is_constrained  --data BlurbGenreCollection --target_sequence dfs```
where:
- --is_constrained: constrained decoding is activated during training and inference.
- --data the dataset name : choices=['WebOfScience', 'BlurbGenreCollection', 'rcv1v2'].
- --target_sequence: the target sequence : choices=['dfs', 'bfs', 'bt_bfs', 'bt_dfs'].
- other parameters like learning rate, batch size, number epochs etc. can be changed.
