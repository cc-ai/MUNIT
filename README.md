**Merging the classes of COCO-Stuff 164k:**

In this branch we perform one main experiment: we merged the labels from CocoStuff-164K according to the following figure:
![cocoStuff_merged_logits](https://raw.githubusercontent.com/cc-ai/MUNIT/feature/cocoStuff_merged_logits/results/merged_coco_classes.png)

To merge labels from a meta-class, we operate on the logits with a softmax function. ([See function](https://github.com/cc-ai/MUNIT/blob/09f1d030959d638ff79ea4b819113b759cee7a55/utils.py#L1131))

**Here are the results of the experiment:**

[See Experiment](https://www.comet.ml/gcosne/synthetic-experiment/2cc5fce671cc4f46b84a39bfeb5f9b1e)

**Results:** 

The network is less constrained than with Cityscapes Classes, it gets back its ability to generate/hallucinate new content and modify the color of the building/sky.
![Results](https://raw.githubusercontent.com/cc-ai/MUNIT/feature/cocoStuff_merged_logits/results/illustration_merge_coco_label.png)
