**What do we do in this branch ?**

In this branch we perform one main experiment: we merged the labels from CocoStuff-164K according to the following figure:
![cocoStuff_merged_logits](https://raw.githubusercontent.com/cc-ai/MUNIT/feature/cocoStuff_merged_logits/results/merged_coco_classes.png)

**Here are the results of the experiment:**

[See Experiment](https://www.comet.ml/gcosne/synthetic-experiment/2cc5fce671cc4f46b84a39bfeb5f9b1e)

**Results:** 

The network is less constrained than with Cityscapes Classes, it gets back its ability to generate/hallucinate new content and modify the color of the building/sky.
![Results](https://raw.githubusercontent.com/cc-ai/MUNIT/feature/cocoStuff_merged_logits/results/illustration_merge_coco_label.png)
