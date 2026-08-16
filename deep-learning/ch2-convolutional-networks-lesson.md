# Chapter 2 — Convolutional Networks: Locality, Equivariance & Scale

Images and many spatial signals have local structure: nearby values interact strongly, and the same pattern can appear in different locations. Convolution encodes these assumptions through local connectivity and shared weights.

## 2.1 From dense to convolutional layers

A dense layer connecting a $224\times224\times3$ image to 1,000 units would need over 150 million weights. A convolutional filter of size $3\times3$ with 3 input and 64 output channels needs

$$
3\cdot3\cdot3\cdot64+64=1{,}792
$$

parameters, reused at every spatial position.

Weight sharing reduces parameters and asserts that a feature is meaningful wherever it occurs.

## 2.2 Discrete convolution

For input $X$ and kernel $K$, deep-learning libraries usually compute cross-correlation:

$$
Y_{i,j,c_{out}}
=b_{c_{out}}+
\sum_{u,v,c_{in}}
K_{u,v,c_{in},c_{out}}X_{i+u,j+v,c_{in}}.
$$

The kernel is not flipped as in mathematical convolution, but learned parameters make the naming difference operationally minor.

Each output channel detects one learned combination across spatial offsets and all input channels.

## 2.3 Output shape

For one spatial dimension with input $N$, kernel $K$, padding $P$, stride $S$, and dilation $D$, output length is

$$
\left\lfloor\frac{N+2P-D(K-1)-1}{S}+1\right\rfloor.
$$

For $N=32$, $K=3$, $P=1$, $S=1$, $D=1$, output remains 32. With stride 2 it becomes 16.

Shape calculations determine residual compatibility and memory use. “Same” padding conventions can differ for even kernels or stride.

## 2.4 Translation equivariance

Ignoring boundaries, shifting the input shifts a convolutional feature map:

$$
f(T_\delta x)=T_\delta f(x).
$$

This is **equivariance**, not invariance. Classification often seeks approximate invariance at the final output, obtained through pooling, aggregation, augmentation, or learned structure.

Padding, stride, aliasing, and finite image boundaries break exact equivariance.

## 2.5 Receptive fields

One $3\times3$ layer sees a local neighbourhood. Stacking layers expands the theoretical receptive field. With stride 1, $L$ such layers yield size

$$
1+2L.
$$

Stride and dilation expand it faster. The **effective** receptive field may be smaller because influence is concentrated near the centre.

Architectures need enough receptive field for the task without discarding fine resolution prematurely.

## 2.6 Stride, pooling, and aliasing

Stride reduces spatial resolution. Max pooling retains the strongest local activation; average pooling retains a local mean. Global average pooling converts each channel map into one value.

Downsampling without low-pass filtering can alias high-frequency patterns. Blur pooling or learned anti-aliasing can improve shift stability. Pooling creates some invariance but loses location detail needed for segmentation and detection.

## 2.7 Dilation and multi-scale structure

Dilated convolution spaces kernel samples apart, expanding receptive field without proportional parameter growth. It helps dense prediction and temporal convolution, but repeated dilation patterns can create gridding artefacts.

Feature pyramids combine multiple resolutions so fine spatial detail and broad semantic context coexist.

## 2.8 Channels and $1\times1$ convolution

A $1\times1$ convolution mixes channels independently at each location. It can expand, compress, or gate channel representations and provides the projection needed when residual branch dimensions change.

Depthwise separable convolution splits spatial filtering per channel from channel mixing, reducing computation from roughly

$$
K^2C_{in}C_{out}
$$

to

$$
K^2C_{in}+C_{in}C_{out}.
$$

Efficiency comes with a changed inductive bias and hardware-dependent speedup.

## 2.9 Residual CNNs

Residual blocks enable deep networks by learning refinements:

$$
y=x+F(x).
$$

Bottleneck blocks use $1\times1$ layers to reduce and restore channels around an expensive spatial convolution. Skip projections align shapes when resolution or channel count changes.

Depth alone does not guarantee better features; optimization, data, augmentation, and compute allocation matter.

## 2.10 Vision training and transfer

Augmentations encode desired invariances: crops, flips, colour transforms, blur, or domain-specific changes. An augmentation that changes the label—such as flipping asymmetric medical anatomy or destroying small financial chart annotations—injects label noise.

Pretrained CNNs provide generic local and mid-level features. Fine-tuning earlier layers with a lower learning rate can adapt them; freezing saves compute but may preserve mismatched features.

## 2.11 Beyond images

One-dimensional convolution models audio, sensor, genomic, and time-series signals. Causal convolution uses only current and past positions. Graph convolution replaces grid neighbourhoods with graph connectivity. The shared idea is structured local aggregation.

For financial time series, naive convolution over a matrix of assets can assume an arbitrary asset ordering. The geometry must have actual meaning.

## 2.12 Failure modes

- Forgetting channel dimensions when counting parameters.
- Calling convolution translation-invariant rather than equivariant.
- Downsampling before preserving task-relevant fine detail.
- Using padding that leaks future values in causal sequences.
- Applying augmentations that change labels or market semantics.
- Assuming theoretical receptive field equals effective influence.
- Comparing FLOPs while ignoring memory movement and hardware kernels.

## 2.13 Knowledge checks

1. Compute parameters in a $5\times5$ convolution from 16 to 32 channels with bias.
2. Distinguish equivariance and invariance.
3. How do stride and dilation affect output and receptive field?
4. Why can downsampling cause aliasing?
5. What assumptions does weight sharing encode?

### Solution outlines

1. $5\cdot5\cdot16\cdot32+32=12{,}832$.
2. Equivariance transforms output consistently with input shift; invariance leaves output unchanged.
3. Stride reduces output resolution and expands jumps; dilation expands kernel coverage without more weights.
4. Frequencies above the new Nyquist limit fold into lower frequencies without prior low-pass filtering.
5. The same local pattern detector is useful across positions.

## 2.14 What to retain

- Convolution encodes locality and shared spatial structure.
- Shapes, receptive fields, and downsampling determine what information can flow.
- Equivariance is architectural; invariance is task-dependent and approximate.
- Residual and channel-mixing designs make deep CNNs trainable and efficient.
- Domain geometry must justify convolutional neighbourhoods.

Next: [Chapter 3 — Recurrent & Sequence Models](ch3-recurrent-sequence-models-viewer.html).
