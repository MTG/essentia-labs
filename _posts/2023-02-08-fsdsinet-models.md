---
layout: post
title: FSD-SINet models released
image: /assets/fsdsinet-models/activations.png
category:
- news
- tensorflow
---

We are delighted to introduce the FSD-SINet models for sound event classification in Essentia.
FSD-SINet is a family of CNN architectures trained in the [FSD50K](https://zenodo.org/record/4060432) dataset with techniques to enhance the shift-invariance properties.
These techniques are:
 - Trainable Low-Pass Filters (TLPF) [1]
 - Adaptive Polyphase Sampling (APS) [2]
 - Intra-Block Pooling (IBP)

The complete list of models supported in Essentia is available on [Essentia's site](https://essentia.upf.edu/models.html#audio-event-recognition) and the experimental results and the implementation details are available in the paper [3] and the official [repository](https://github.com/edufonseca/shift_sec).
According to the authors, the [fsd-sinet-vgg42-tlpf-aps-1](https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf-ibp-1.pb) model featuring TLPF and APS obtained the best evaluation metrics.
Additionally, [fsd-sinet-vgg41-tlpf-ibp-1](https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg41-tlpf-ibp-1.pb) is a lighter architecture using TLPF and IBP intended for reduced computational cost.

As an example, let's analyze a field recording and observe the model's predictions.

<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/636921/simple/large/" width="760" height="245"></iframe>

The following code performs inference with the `fsd-sinet-vgg42-tlpf-aps-1` model:

```python
from essentia.standard import MonoLoader, TensorflowPredictFSDSINet

audio_file = "636921__girlwithsoundrecorder__sounds-from-the-forest.wav"
audio = MonoLoader(filename=audio_file, sampleRate=22050)()
model = TensorflowPredictFSDSINet(graphFilename="fsd-sinet-vgg42-tlpf-aps-1.pb")
activations = model(audio)
```
> *Note*: Remember to update Essentia before running the code.

By default, these model predict sound event activations with a rate of 2 Hz.
We can visualize the top activations using `matplotlib`:

```python
import json
import matplotlib.pyplot as plt
import numpy as np


def get_top_indices(data, top_n=10):
    av = np.mean(data, axis=1)
    sorting = np.argsort(av)[::-1]
    return sorting[:top_n], [av[i] for i in sorting]

# Read the moetadata
metadata_file = "fsd-sinet-vgg41-tlpf-ibp-1.json"
metadata = json.load(open(metadata_file, "r"))
labels = metadata["classes"]

# Compute the top-n labels and predictions
top_n, averages = get_top_indices(predictions, top_n=15)
top_labels = [labels[i] for i in top_n]
top_labels_with_av = [f"{label} ({av:.3f})" for label, av in zip(top_labels, averages)]

top_predictions = np.array([predictions[i, :] for i in top_n])

# Generate plots and improve formatting
matfig = plt.figure(figsize=(8, 3))
plt.matshow(top_predictions, fignum=matfig.number, aspect="auto")

plt.yticks(np.arange(len(top_labels_with_av)), top_labels_with_av)
locs, _ = plt.xticks()
ticks = np.array(locs // 2).astype("int")
plt.xticks(locs[1: -1], ticks[1: -1])
plt.tick_params(
    bottom=True,
    top=False,
    labelbottom=True,
    labeltop=False,
)
plt.xlabel("(s)")

plt.savefig("activations.png", bbox_inches='tight')
```

![png]({{ site.baseurl }}/assets/fsdsinet-models/activations.png)

As it can be seen, the model detects the intermittent bird sounds and some of the breathing and step sounds of the person recording.

## References

[1] R. Zhang, “Making convolutional networks shift-invariant again,” in International Conference on Machine Learning. PMLR, 2019, pp. 7324–7334.

[2] A. Chaman and I. Dokmanic, “Truly shift-invariant convolutional neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 3773–3783.

[3] Eduardo Fonseca, Andres Ferraro, Xavier Serra, "Improving Sound Event Classification by Increasing Shift Invariance in Convolutional Neural Networks", arXiv:2107.00623, 2021.

[4] Audio by Freesound user [GirlWithSoundRecorder](https://freesound.org/people/GirlWithSoundRecorder/sounds/636921/).
