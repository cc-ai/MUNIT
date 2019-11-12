#  Introduction

<p style="text-align:justify;" >
    Dramatic and rapid changes to the global economy are required in order to limit climate-related risks for natural and human systems (IPCC, 2018). <b>Governmental interventions are needed to fight climate change</b> and they need strong public support.  However,  <b>it is difficult to mentally simulate the complex effects of climate change</b> (O’Neill & Hulme, 2009) and people often discount the impact that their actions will have on the future, especially if the <b>consequences are long-term</b>, <b>abstract</b>, and at odds with current behaviors and identities (Marshall, 2015).
</p> 

- We are developing a tool to **help the public understand the consequences of climate change.** 

- We intend to **make people aware of Climate Change** **in their direct environment** by showing them concrete examples. 

<p style="text-align:justify;">Currently we are focusing on simulating images of one specific extreme climate event: floods. We are aiming to create a flood simulator which, given a user-entered address, is able to extract a street view image of the surroundings and to alter it to generate a plausible image projecting flood where it is likely to occur.</p>
<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/flooding2.gif" style="zoom:100%;" alt="{{ include.description }}" class="center"> 
  <figcaption>Visualization made with a Generative Adversarial Network (GAN) </figcaption> 
</figure>
</div>



<p style="text-align:justify;">Recent research has explored the <b>potential of translating numerical climate models into representations</b>  that are intuitive and easy to understand, for instance via <b>climate-analog mapping</b>  (Fitzpatrick et al., 2019) and by leveraging relevant social group norms (van der Linden, 2015). Other approaches have focused on selecting relevant images to best represent climate change impacts (Sheppard, 2012; Corner & Clarke, 2016) as well as using artistic renderings of possible future landscapes (Giannachi, 2012) and even video games (Angel et al., 2015). However, to our knowledge, our project is the <b>first application of generative models to generate images of future climate change scenarios.</b>  </p>
# Technical Proposal

<p style="text-align:justify;">We propose to use <b>Style Transfer</b> and especially Unsupervised <b> Image To Image Translation techniques</b> to learn a transformation from a natural image of a house to its flooded version.  This technology can leverage the quantity of cheap-to-acquire unannotated images.  </p>
> **Image To Image Translation**
> :   A class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs
>
> **Style Transfer**
> :   Aims to modify the style of an image while preserving its content.
> [																																	CycleGAN (Zhu et al., 2017)](https://junyanz.github.io/CycleGAN/) 

<p style="text-align:justify;">Let <a href="https://www.codecogs.com/eqnedit.php?latex=x_1&space;\in&space;X_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_1&space;\in&space;X_1" title="x_1 \in X_1" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=x_2&space;\in&space;X_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_2&space;\in&space;X_2" title="x_2 \in X_2" /></a>  be images from two different image domains. <a href="https://www.codecogs.com/eqnedit.php?latex=X_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_1" title="X_1" /></a> represents the  non-flooded domain which gathers several type of street-level imagery defined later in the data section and <a href="https://www.codecogs.com/eqnedit.php?latex=X_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_2" title="X_2" /></a> is the flooded domain composed of images where a part of a single house or building is visible and the street is partially or fully covered by water. </p>
<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_1.png" style="zoom:38%;" alt="{{ include.description }}" class="center"> 
  <figcaption><b>Non-flood sample:</b>&nbsp;&nbsp;&nbsp;<a href="https://www.codecogs.com/eqnedit.php?latex=x_1&space;\in&space;X_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_1&space;\in&space;X_1" title="x_1 \in X_1" /></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <b>Flood sample:</b>&nbsp;&nbsp;&nbsp;<a href="https://www.codecogs.com/eqnedit.php?latex=x_2&space;\in&space;X_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_2&space;\in&space;X_2" title="x_2 \in X_2" /></a> </figcaption> 
</figure>
</div>

<p style="text-align:justify;">In the unpaired image-to-image translation setting, we are given samples drawn from two marginal distributions : <img src="https://latex.codecogs.com/gif.latex?$x_1&space;\sim&space;p(x_1)$" title="$x_1 \sim p(x_1)$" />samples of (non-flooded) houses and <img src="https://latex.codecogs.com/gif.latex?$x_2&space;\sim&space;p(x_2)$" title="$x_2 \sim p(x_2)$" /> samples of flooded houses, without access to the joint distribution <img src="https://latex.codecogs.com/gif.latex?$p(x_1,x_2)$" title="$p(x_1,x_2)$" />.</p>
<p style="text-align:justify;">From a probability theory viewpoint, the key challenge is to learn the joint distribution while only observing the marginals. Unfortunately, there is an infinite set of joint distributions that correspond to the given marginal distributions <a href="https://en.wikipedia.org/wiki/Coupling_(probability)">(cf coupling theory)</a>. Inferring the joint distribution from the marginals is a highly ill-defined problem. Assumptions are required to constrain the structure of the joint distribution, such as those introduced by the authors of CycleGAN. </p>
<p style="text-align:justify;">In our case we are estimating the complex conditional distribution <img src="https://latex.codecogs.com/gif.latex?$p(x_2|x_1)$" title="$p(x_2|x_1)$" /> with different image-to-image translation models <img src="https://latex.codecogs.com/gif.latex?$p(x_{1\rightarrow&space;2}|x_1)$" title="$p(x_{1\rightarrow 2}|x_1)$" />, where <img src="https://latex.codecogs.com/gif.latex?$x_{1\rightarrow&space;2}$" title="$x_{1\rightarrow 2}$" /> is a sample produced by translating <img src="https://latex.codecogs.com/gif.latex?$x_1$" title="$x_1$" /> to <img src="https://latex.codecogs.com/gif.latex?$X_2$" title="$X_2$" />. </p>
## CycleGAN 

<p style="text-align:justify;">CycleGAN is one of the research papers that revolutionized image-to-image translation in an unpaired setting. It has been used as the first proof of concept for this project. </p>
<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_2.png" style="zoom:40%;" alt="{{ include.description }}" class="center"> 
  <figcaption> Results of the proof of concept from <a href="https://arxiv.org/abs/1905.03709">Schmidt et al., 2019</a> </figcaption> 
</figure>
</div>

It aims to capture the style from one image collection and to learn how to apply it to the other image collection. There are two main constraints in order to ensure conversion and coherent transformation:

<p style="text-align:justify;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The <b>indistinguishable</b> constraint: The produced output has to be indistinguishable from the samples of the new domain. This is enforced using the GAN Loss <a href="https://arxiv.org/abs/1406.2661">Goodfellow et al., 2014</a> and is applied at the distribution level. In our case, the mapping of the non-flood domain to the flood domain should create images that are indistinguishable from the training images of floods and vice-versa <a href="https://arxiv.org/abs/1709.00074">Galenti et al., 2017</a>. But this constraint alone is not enough to map an input image in domain <a href="https://www.codecogs.com/eqnedit.php?latex=X_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_1" title="X_1" /></a> to an output image  <a href="https://www.codecogs.com/eqnedit.php?latex=X_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_2" title="X_2" /></a>  with the same semantics. The network could learn to generate realistic images from domain  <a href="https://www.codecogs.com/eqnedit.php?latex=X_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_2" title="X_2" /></a> without preserving the content of the input image. This latter point is tackled by the cycle consistency constraint.</p> 
<p style="text-align:justify;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The <b> Cycle-Consistency Constraint</b>  aims to regularize the mapping of the two domains in a <i> meaningful way </i>. It can be described as imposing a structural constraint which states that if we translate from one domain to the other and back again we should arrive at where we started. Formally, if we have a translator <img src="https://latex.codecogs.com/gif.latex?$G:X_1&space;\rightarrow&space;X_2$" title="$G:X_1 \rightarrow X_2$" /> and another translator <img src="https://latex.codecogs.com/gif.latex?$F:X_2&space;\rightarrow&space;X_1$" title="$F:X_2 \rightarrow X_1$" /> then <img src="https://latex.codecogs.com/gif.latex?$G$" title="$G$" /> and <img src="https://latex.codecogs.com/gif.latex?$F$" title="$F$" />should be bijections and inverses of each other. </p>
<p style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/scheme_cycle_consistency.png" style="zoom:55%;" alt="{{ include.description }}" class="center"> 
  <figcaption> </figcaption> </p>

- <p style="text-align:justify;">(a) The CycleGAN model contains two mapping functions <img src="https://latex.codecogs.com/gif.latex?G:X_1&space;\rightarrow&space;X_2" title="G:X_1 \rightarrow X_2" /> and <img src="https://latex.codecogs.com/gif.latex?F:X_2&space;\rightarrow&space;X_1" title="$F:X_2 \rightarrow X_1$" /> , and associated adversarial discriminators <img src="https://latex.codecogs.com/gif.latex?$D_{X_2}$" title="$D_{X_2}$" /> and <img src="https://latex.codecogs.com/gif.latex?$D_{X_1}$" title="$D_{X_1}$" />. <img src="https://latex.codecogs.com/gif.latex?$D_{X_2}$" title="$D_{X_2}$" /> encourages <img src="https://latex.codecogs.com/gif.latex?$G$" title="$G$" /> to translate <img src="https://latex.codecogs.com/gif.latex?X_1" title="X_1" /> into outputs indistinguishable from domain <img src="https://latex.codecogs.com/gif.latex?X_2" title="$X_2$" />, and vice versa for <img src="https://latex.codecogs.com/gif.latex?$D_{X_1}$" title="$D_{X_1}$" /> and <img src="https://latex.codecogs.com/gif.latex?$F$" title="$F$" />. </p>

- <p style="text-align:justify;">(b) Forward cycle-consistency loss: <img src="https://latex.codecogs.com/gif.latex?$x_1\rightarrow&space;G(x_1)&space;\rightarrow&space;F(G(x_1))&space;\approx&space;x_1$" title="$x_1\rightarrow G(x_1) \rightarrow F(G(x_1)) \approx x_1$"/> </p>

- <p style="text-align:justify;">(c) Backward cycle-consistency loss: <img src="https://latex.codecogs.com/gif.latex?$x_2&space;\rightarrow&space;F(x_2)&space;\rightarrow&space;G(F(x_2))&space;\approx&space;x_2&space;$" title="$x_2 \rightarrow F(x_2) \rightarrow G(F(x_2)) \approx x_2 $" /> </p>

<p style="text-align:justify;"><b>Pros and cons:</b> The most advantageous part of this approach is its total lack of supervision, which means that the access to data is cheap (1K images of non-flooded and flooded houses). The major problem is that the style transfer is applied to the entire image.</p>
<p style="text-align:justify;"><b>Initial Results:</b> When the ground is not concrete but grass and vegetation, CycleGAN generates a brown flood of low quality with blur on the edges between houses and grass. The color of the sky changes from blue to grey (probably because of the bias on the training set of flood images). </p>
## InstaGAN


<p style="text-align:justify;">The <a href="https://arxiv.org/abs/1406.2661">InstaGAN</a> architecture is built on the foundations of CycleGAN. The main idea of their approach is to incorporate instance attributes <img src="https://latex.codecogs.com/gif.latex?\mathcal{A}" title="\mathcal{A}" /> (and <img src="https://latex.codecogs.com/gif.latex?\mathcal{B}" title="\mathcal{B}" />) to the source <img src="https://latex.codecogs.com/gif.latex?X_1" title="X_1" /> (and the target <img src="https://latex.codecogs.com/gif.latex?X_2" title="X_2" />) domain to improve the image-to-image translation. They describe their approach as learning joint mappings between attribute-augmented spaces <img src="https://latex.codecogs.com/gif.latex?X_1" title="X_1" /> × <img src="https://latex.codecogs.com/gif.latex?\mathcal{A}" title="\mathcal{A}" /> and <img src="https://latex.codecogs.com/gif.latex?X_2" title="X_2" /> × <img src="https://latex.codecogs.com/gif.latex?\mathcal{B}" title="\mathcal{B}" />. </p>
<p style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_3.png" style="zoom:55%;" alt="{{ include.description }}" class="center"> 
  <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Overview of the Network &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Generator &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Discriminator </figcaption> </p>

<p style="text-align:justify;">In our setting, the set of instance attributes <img src="https://latex.codecogs.com/gif.latex?a\in\mathcal{A}" title="a\in\mathcal{A}" />  is reduced to one attribute: a segmentation mask of <b>where-to-flood</b>  and for the attribute of <img src="https://latex.codecogs.com/gif.latex?b\in\mathcal{B}" title="b\in\mathcal{B}" /> a segmentation mask covering the <b>flood</b>. Each network is designed to encode both an image and a set of masks (in our case a single mask). </p>
<p style="text-align:justify;">The authors explicitly say that any useful information could be incorporated as an attribute and claim that their approach leads to disentangle different instances within the image allowing the generator to perform <b>accurate</b> and <b>detailed</b> translation from one instance to the other. </p>
<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_instagan.png" style="zoom:55%;" alt="{{ include.description }}" class="center"> 
  <figcaption>Best-case scenario: failure cases &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</figcaption> 
</figure>
</div>

<p style="text-align:justify;"><b>Pros and cons:</b>. As for CycleGAN InstaGAN doesn't need paired images but requires the knowledge of some attributes, here the masks. Sometimes the model is able to render water in a realistic manner, including reflections and texture. But a major drawback is that, although it's penalized during training, it continues to modify the rest of the image (the unmasked region): colors change, artifacts appear, textures are different and fine details are blurred. </p>
<p style="text-align:justify;"> <b>Results:</b> Empirically we find that it works well with grass but not with concrete. Transparency is a big issue with InstaGAN's results on our task, since most of the time we can see the road lanes through the flood. Even in synthetic settings with aligned images InstaGAN generates relatively realistic water texture which remains transparent. We could conclude that it learns to reflect the sky on the water (whatever the color of the sky is), resulting in the fact that sometimes it paints blue on the concrete itself without the accompanying water texture. In our case results quality worsen dramatically out of the training set.</p>
<p style="text-align:justify;"><b>Note:</b> The instances used in the papers are either segmentation mask of animals (e.g. translating sheep to giraffe), or segmentation mask of clothes (e.g. translating pants to skirt). In both cases, I found that theses instances are <em>less diverse</em> than instances from our non-flood to flood translation in the sense that sheep <em> color, shape, texture </em> is less diverse than the examples of flood or street in our dataset. </p>
## Generative Image Inpainting

<p style="text-align:justify;">Previous approaches based on modification of CycleGAN does not give us a fine control over the region that should be flooded. Assuming we are able to identify such a region in the image, we would only need to learn how to render water realistically. There are a lot of promising image edition techniques in the GAN literature demonstrating how to perform edition of specific attributes, morphing images or manipulating the semantic. These transformation are often performed on small latent space of generated fake images.  However, natural image edition is a lot harder and there is no easy way of manipulating the semantic of natural images. </p>
<p style="text-align:justify;"><b>Image Inpainting</b> is the technique of modifying and restoring a damaged image in a visually plausible way. Given recent advances in the field, it is now possible to <em>guess</em> lost information and replace it with plausible content at real-time speed. </p>
<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_inpainting.png" style="zoom:70%;" alt="{{ include.description }}" class="center"> 
  <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Destructed Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Reconstructed Image with DeepFill</figcaption> 
</figure>
</div>

<p style="text-align:justify;">For example, a recent deep-generative model exploiting contextual attention: <a href="http://jiahuiyu.com/deepfill/">DeepFill</a>, is able to reconstruct high definition altered images of faces and landscapes at real-time speed. We believe that there is a way of leveraging the network generation capacity and apply its mechanisms to our case. Our experiment consist in biasing DeepFill to reconstruct only region where there is water (without surrounding water).  We trained the network with several hundreds images of flood where the water was replaced by a grey mask. At inference we replaced what we defined as the ground with a grey mask. (see results below) </p>
<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_5.png" style="zoom:55%;" alt="{{ include.description }}" class="center"> 
  <figcaption>Deepfill biased to replace a given mask (here the ground) with plausible water. </figcaption> 
</figure>
</div>

<p style="text-align:justify;"> <b>Results:</b> The quality of the result is bad when given large masks. This could be explain by the fact that the architecture is designed to extract information from the context in the image: in the former experiment the network had to <em>draw</em> from a context where water is inexistent. To pursue research in that direction, one may want to give a better context to the network by using example of water texture on the side of the image. Or by increasing the dataset of images where there is water. </p>
# Current Approach

<p style="text-align:justify;"> Our current approach is built on <a href="https://arxiv.org/abs/1804.04732">MUNIT</a>. In the paper, a <em>partially shared latent space assumption</em> is made. It is assumed that images can be disentangled into a content code (domain-invariant) and a style code (domain-dependant). In this assumption, each image <img src="https://latex.codecogs.com/gif.latex?x_{i}\in\mathcal{X}_{i}" title="x_{i}\in\mathcal{X}_{i}" /> is generated from a content latent code <img src="https://latex.codecogs.com/gif.latex?c\in&space;\mathcal{C}" title="c\in \mathcal{C}" /> that is shared by both domains, and a style latent code <img src="https://latex.codecogs.com/gif.latex?s_{i}\in&space;\mathcal{S}_{i}" title="s_{i}\in \mathcal{S}_{i}" /> that is specific to the individual domain. </p>


<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_munit.png" style="zoom:45%;" alt="{{ include.description }}" class="center"> 

  <figcaption>(a) Overview of MUNIT Autoencoder&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) Off the shelf model trained on our dataset.</figcaption> 
</figure>
</div>



<p style="text-align:justify;">In other words, a pair of corresponding images <img src="https://latex.codecogs.com/gif.latex?(x_{1},x_{2})" title="(x_{1},x_{2})" /> from the joint distribution is assumed to be generated by <img src="https://latex.codecogs.com/gif.latex?x_{1}&space;=&space;G^{*}_{1}(c,&space;s_{1})" title="x_{1} = G^{*}_{1}(c, s_{1})" /> and <img src="https://latex.codecogs.com/gif.latex?x_{2}&space;=&space;G^{*}_{2}(c,&space;s_{2})" title="x_{2} = G^{*}_{2}(c, s_{2})" />, where <img src="https://latex.codecogs.com/gif.latex?c,&space;s_{1},&space;s_{2}" title="c, s_{1}, s_{2}" /> are from some prior distributions and <img src="https://latex.codecogs.com/gif.latex?G^{*}_{1}" title="G^{*}_{1}" />, <img src="https://latex.codecogs.com/gif.latex?G^{*}_{2}" title="G^{*}_{2}" /> are the underlying generators. Given the former hypothesis, the goal is to learn the underlying generator and encoder functions
with neural networks.</p>

<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_itit_munit.png" style="zoom:100%;" alt="{{ include.description }}" class="center">  <figcaption>MUNIT image-to-image translation model consists of two auto-encoders (denoted by <font color="red">red</font> and <font color="blue">blue</font> arrows respectively), one for each domain. The latent code of each auto-encoder is composed of a content code <img src="https://latex.codecogs.com/gif.latex?c" title="c" /> and a style code <img src="https://latex.codecogs.com/gif.latex?s" title="s" />.</figcaption> 
</figure>
</div>

<p style="text-align:justify;"><b>Image-to-image translation</b> is performed by swapping encoder-decoder pairs. For example, to translate a house  <img src="https://latex.codecogs.com/gif.latex?x_{1}\in&space;\mathcal{X}_{1}" title="x_{1}\in \mathcal{X}_{1}" />  to a flooded-house <img src="https://latex.codecogs.com/gif.latex?\mathcal{X}_{2}" title="\mathcal{X}_{2}" />, one may use MUNIT to first extract the content latent code <img src="https://latex.codecogs.com/gif.latex?c_{1}&space;=&space;E^{c}_{1}(x_{1})" title="c_{1} = E^{c}_{1}(x_{1})" /> of the house image that we want to flood and randomly draw a style latent code <img src="https://latex.codecogs.com/gif.latex?s_{2}" title="s_{2}" /> from the prior distribution <img src="https://latex.codecogs.com/gif.latex?q(s_{2})\sim&space;\mathcal{N}(0,&space;\mathbf{I})" title="q(s_{2})\sim \mathcal{N}(0, \mathbf{I})" /> of flooded-houses and then use <img src="https://latex.codecogs.com/gif.latex?G_{2}" title="G_{2}" /> to produce the final output image <img src="https://latex.codecogs.com/gif.latex?x_{1\rightarrow&space;2}&space;=&space;G_{2}(c_{1},&space;s_{2})" title="x_{1\rightarrow 2} = G_{2}(c_{1}, s_{2})" />(content from <img src="https://latex.codecogs.com/gif.latex?x_1" title="x_1" />and style from <img src="https://latex.codecogs.com/gif.latex?x_2" title="x_2" />). </p>
### How does it work ?

<p style="text-align:justify;"> <a href="https://arxiv.org/abs/1703.06868">Huang et al.</a>  demonstrated that Instance Normalization is deeply linked to style normalization. Munit transfer the style by modifying the features statistics, de-normalizing in a certain way.  Given an input batch <img src="https://latex.codecogs.com/gif.latex?x&space;\in&space;\mathbb{R}^{N\times&space;C\times&space;H\times&space;W}" title="x \in \mathbb{R}^{N\times C\times H\times W}" />, <b>Instance Normalization Layers</b> are used in MUNIT encoders to normalize feature statistics. </p>
<p style="text-align:justify;"><img src="https://latex.codecogs.com/gif.latex?IN(x)&space;=&space;\left&space;(&space;\frac{x-\mu(x)}{\sigma(x)}&space;\right&space;)" title="IN(x) = \gamma \left ( \frac{x-\mu(x)}{\sigma(x)} \right ) " /> </p>
<p style="text-align:justify;">Where <img src="https://latex.codecogs.com/gif.latex?\mu(x)" title="\mu(x)" /> and <img src="https://latex.codecogs.com/gif.latex?\sigma(x)" title="\sigma(x)" />are computed as the mean and standard deviation across spatial dimensions independently for each channel and each sample. <b>Adaptative Instance normalization layers </b> are then used in the decoder to de-normalize the features statistics.  </p> 
<p style="text-align:justify;"><img src="https://latex.codecogs.com/gif.latex?AdaIN(z,\gamma,\beta)&space;=&space;\gamma&space;\left&space;(&space;\frac{z-\mu(z)}{\sigma(z)}&space;\right&space;)&space;&plus;&space;\beta" title="AdaIN(z,\gamma,\beta) = \gamma \left ( \frac{z-\mu(z)}{\sigma(z)} \right ) + \beta" /> </p>
<p style="text-align:justify;">With <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /> and  <img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" />  defined as a multi-layer perceptron (MLP), i.e., [<img src="https://latex.codecogs.com/gif.latex?\beta;\gamma" title="\beta;\gamma" />] = [<img src="https://latex.codecogs.com/gif.latex?\beta(s);\gamma(s)" title="\beta(s);\gamma(s)" />]=MLP<img src="https://latex.codecogs.com/gif.latex?(s)" title="(s)" />with  <img src="https://latex.codecogs.com/gif.latex?s" title="s" /> the style. The fact that the de-normalization parameters are inferred with a MLP allow users to generate multiple output from one image. </p>
### Modifying the network to fit our purpose:

We questioned and transformed the official MUNIT architecture to fit our purpose. 

- <p style="text-align:justify;">Because we wanted control over the translation, we removed randomness from the style: the network is then trained to perform style transfer with the style extracted from one image and not sampled from a normal distribution. </p>

- After analyzing the feature space of the style, [T-SNE plot](https://docs.google.com/document/d/1rNVtQL071r6J8sj0mV0zDaNAnrczTJebBMsZwmm08to/edit?usp=sharing) we decided that sharing the weights between the style encoders could help the network to extract informative features. Since the results were not affected by this ablation we kept it. ([See Experiment](https://www.comet.ml/gcosne/munit-one-style/b42076bac551495d86bda1d1011e11bc?experiment-tab=images))

- We shrink the architecture to use a single AutoEncoder and concluded that it was either longer to converge or that the transformation was harder to learn since the results were affected negatively. ([See Experiment](https://www.comet.ml/gcosne/munit-uni-encoder/7cd0051ad112466190bb6d31f7291cc7))

- Based on the fact that the flooding process is destructive and that there is no reason that the network could reconstruct the road from the flooded version, we implemented a weaker version of the Cycle Consistency Loss where the later is only computed on a specific region of the image. The specific region is defined by a binary mask of where we think the image should be altered. For example a flooded image mapped to a non-flooded house should only be altered in an area close by the one delimited by the water. (In practice there are bias intrinsic to the dataset such as the sky often being gray in a image of flood) ([See Experiment][https://www.comet.ml/gcosne/munit-v2/3ee5653a971a459486edc45c12a0ae22?experiment-tab=images])

- We trained a [classifier][https://github.com/cc-ai/floods-gans/tree/master/flood-classifier] to distinguish between flooded and non-flooded images (binary output) then use it when training MUNIT with a Loss on the generator indicating that fake flooded (resp non-flooded) images should be identified as flooded (resp non-flooded) by the classifier. It didn't improve the results we had, like if the flood classifier was a very bad discriminator that the generator could trick easily. ([See Experiment][https://www.comet.ml/gcosne/munit-one-style-classifier/9bc7d50408ab4393a09acaadcda58e6a?experiment-tab=images])

- To push the style encoder towards learning meaningful information, we investigated how to anonymise the representation of the content feature learned by MUNIT encoder. The idea behind is that if the content feature doesn't contains information about the domain it has been encoded, then the style would encode this information. We hence minimized the mutual information between the <em>content</em> feature and the source of the <em>content</em>. To do so we used a Domain-Classifier as in [Learning Anonymized Representations with Adversarial Neural Networks](https://arxiv.org/abs/1802.09386)

- We experiment playing with the training ratio of the Discriminator and the Generator. We empirically found that a factor 5 does improve slightly the convergence speed.

<b>Major Changes:</b> Introducing a Semantic Consistency Loss, we use [DeepLab v2](https://github.com/cc-ai/floods-gans/tree/master/ground_segmentation) trained on cityscape to infer semantic label, and implemented an additional loss indicating to the generator that every fake image should keep the same Semantic as the source image before translation everywhere except a defined region where we think there should be an alteration. This modification dramatically improved our results. ([See Experiment][https://www.comet.ml/gcosne/munit-v2-semantic-seg/68a929738a20413f885d2d684d05bd47?experiment-tab=images]) 

We also experimented with [DeeplabV2][https://github.com/kazuto1011/deeplab-pytorch] trained on [COCO-Stuff](https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/cocostuff/README.md). We thought this version would better suit our problem because it is able to identify water on the road but it turned out that (maybe because of the large number of classes) it didn't constrained much the network as with the previous version. We also tried to merge the classes from coco-stuff to only keep meta-classes that would be similar to cityscapes, it would allow us to keep a small number of classes and leverage the potential of identifying the water. (Impossible with Cityscape classes) ([See Results][https://github.com/cc-ai/MUNIT/blob/feature/cocoStuff_merged_logits/README.md])

## How to leverage simulated data ?

<p style="text-align:justify;">We plan of using a simulated world built by Vahe Vardanyan with the Graphics Engine Unity to simulate different types of houses and streets under flood conditions to help our GAN understand where it should flood. </p>
<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_synthetic.png" style="zoom:55%;" alt="{{ include.description }}" class="center"> 
  <figcaption><b>Non-flood Synthethic sample</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <b>Flood Synthetic sample</b>&nbsp;&nbsp;&nbsp; </figcaption> 
</figure>
</div>

<p style="text-align:justify;">One main advantage of using synthetic data is that theoretically we would have access to an unlimited amount of pairs. The principal difficulty lies in leveraging those pairs despite the existing discrepancy between the distribution of synthetic and real data. </p>
<div style="text-align: center">
<figure class="image"> 
  <img src="https://raw.githubusercontent.com/cc-ai/MUNIT/master/results/illustration_tsne.png" style="zoom:100%;" alt="{{ include.description }}" class="center"> 
  <figcaption></figcaption> 
</figure>
</div>

<p style="text-align:justify;">We can visualize the discrepancies between the differents domains with a T-SNE plot. Learning to flood natural images is equivalent to adapt samples from domain <img src="https://latex.codecogs.com/gif.latex?X_1" title="X_1" /> to domain <img src="https://latex.codecogs.com/gif.latex?X_2" title="X_2" /> and we would like to help the network learn this translation with an <em> easier task</em>: translating from <img src="https://latex.codecogs.com/gif.latex?X_{1\_Synthetic}" title="X_1" /> to <img src="https://latex.codecogs.com/gif.latex?X_{2\_Synthetic}" title="X_1" />. Indeed, probably because of its pairs, the gap separating the synthetic domains is smaller than for the real one. We also notice that some of the real data are mixed with the synthetic cluster, somehow a proof that the synthetic world is well imitating the real world.</p> 
<p style="text-align:justify;">We mix simulated data with their natural equivalent (synthetic flooded images with flooded images) at training time with an additional pixelwise reconstruction loss computed on the pixel that shouldn't be altered. </p>
<p style="text-align:justify;"><img src="https://latex.codecogs.com/gif.latex?\large&space;\mathcal{L}_{synthetic}(x_1,x_{1\rightarrow&space;2})&space;=&space;\left&space;|$mask$\cdot(x_1$-$x_{1\rightarrow&space;2})&space;|&space;\right&space;|" title="\large \mathcal{L}_{synthetic}(x_1,x_{1\rightarrow 2}) = \left |$mask$\cdot(x_1$-$x_{1\rightarrow 2}) | \right |" /> </p>

<p style="text-align:justify;">Where <b>mask</b>  <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\large&space;$mask$&space;=&space;(x_1==x_2)" title="\large $mask$ = (x_1==x_2)" /> correspond to the region of pixels where <img src="https://latex.codecogs.com/gif.latex?x_1" title="x_1" /> and <img src="https://latex.codecogs.com/gif.latex?x_2" title="x_1" /> are paired, in our case, where there is no water.</p>

## Evaluating the realism of our results 

<p style="text-align:justify;">We synthesized our attempt to establish an automated evaluation metric to quantify fake Image  realism in the following <a href="https://arxiv.org/abs/1910.10143">paper</a> . Our work consisted in adapting several existing metrics (<a href="https://arxiv.org/abs/1801.01973">IS</a>,<a href="https://arxiv.org/abs/1706.08500">FID</a>, <a href="https://arxiv.org/abs/1801.01401">KID</a>..) and assessing them against gold standard human evaluation: <a href="https://arxiv.org/abs/1904.01121">HYPE</a>. While insufficient alone to establish a human-correlated automatic evaluation metric, we believe this work begins to bridge the gap between human and automated generative evaluation procedures.</p>

## Data Mining And Annotation: 

We set a goal of recovering about 1000 images in each domain meeting a number of criteria.

***Flooded Houses***: images should present a part of a single house or building visible and the street partially or fully covered by water.

These images have been gathered using the results of different Google Image queries focusing on North-American suburban type of houses. 

***Non-flooded houses are a mix of several types of images:*** 

- Single houses with grass gathered manually from the Web.
- Street-level imagery extracted from Google StreetView API.
- Diverse street-level imagery covering a variety of weather conditions, seasons, times of day  and viewpoints taken from a publicly available dataset.

Motivated by the idea that it would be easier to perform Image To Image Translation if our GAN had an idea of what the concepts of **Ground** and **Water** are, we increased the knowledge we had on the dataset by annotating pixels corresponding to **Water** in the Flooded Houses images and those corresponding to the **Ground** in Non-flooded houses images.

- **70%** of the Flooded Houses were annotated using a Semantic Segmentation Network, namely  [DeepLab v2 trained on COCO-stuffs-164k dataset](https://github.com/cc-ai/deeplab-pytorch) and merging some labels to output a binary mask of water.

- **30%** of them have been manually annotated using [LabelBox](https://labelbox.com/).
- **100%** of the Non Flooded Houses were automatically segmented using [DeepLab trained on CityScapes](https://github.com/cc-ai/floods-gans/blob/master/ground_segmentation/README.md).