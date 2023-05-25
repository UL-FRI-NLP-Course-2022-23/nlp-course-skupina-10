# Data Augmentation

We apply various data augmentation techniques to enhance the size of the training set. The following transformations are utilized:

1. **Paraphrase swapping**: Each paraphrase pair `[p1, p2]` is used twice, enabling the model to learn mappings from `p1 -> p2` and `p2 -> p1`. 

2. **Paraphrase duplication**: A dataset is constructed in the form of `[pi, pi, 0]`, where `pi` represents some sentence and `0` indicates that the pair is not a paraphrase. This can be usefull for models such as bart-paraphrase, which utilize a `is paraphrase / is not paraphrase` label.

3. **Synonym replacement**: Random synonyms in sentences are replaced to introduce lexical variations. Initially, we explored automatic techniques using WordNet, but found them to be too noisy. Instead, we manually replaced synonyms to ensure high-quality data augmentation.
