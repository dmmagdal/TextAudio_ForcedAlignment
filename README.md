# Text/Audio Forced Alignment

Description: This repo is a designed to learn about different ways to do forced alignment between text and audio. This can be applied to Automatic Speech Recognition (ASR) (aka Speech to Text (STT)) or Text to speech (TTS) applications.


### Notes


### References

 - Wav2Vec 2.0
     - Wav2Vec: Unsupervised pre-training for speech recognition [Arxiv paper](https://arxiv.org/pdf/1904.05862.pdf)
     - Wav2Vec: Unsupervised pre-training for speech recognition [YouTube video](https://www.youtube.com/watch?v=XkUVOijzAt8&ab_channel=MLOpsGuru)
     - The Illustrated Wav2Vec [Blog post](https://jonathanbgn.com/2021/06/29/illustrated-wav2vec.html)
     - Wav2Vec 2.0: A Framework for self-supervised Learning of Speech Representations [Arxiv paper](https://arxiv.org/pdf/2006.11477.pdf)
     - Wav2Vec 2.0: A Framework for self-supervised Learning of Speech Representations [YouTube video](https://www.youtube.com/watch?v=aUSXvoWfy3w&ab_channel=MLOpsGuru)
     - An Illustrated Tour of Wav2Vec 2.0 [Blog post](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)
     - Facebook Research fairseq [GitHub repo](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) (Wav2Vec 2.0 README.md)
     - Forced Alignment with Wav2Vec 2 [Pytorch tutorial](https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html)
     - Fine-tuning Wav2vec2 with an LM Head (for ASR) [Tensorflow tutorial](https://www.tensorflow.org/hub/tutorials/wav2vec2_saved_model_finetuning)
     - Tensorflow Hub Wav2Vec 2.0 [model checkpoints](https://tfhub.dev/s?q=wav2vec2):
         - [Wav2Vec2](https://tfhub.dev/vasudevgupta7/wav2vec2/1)
         - [Wav2Vec2-960h](https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1)
     - Huggingface Wav2Vec 2.0:
         - [Model page/documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
         - Huggingface Hub models:
            - Facebook wav2vec2-base [model](https://huggingface.co/facebook/wav2vec2-base) (Torch)
            - Facebook wav2vec2-large [model](https://huggingface.co/facebook/wav2vec2-large) (Torch)
            - Facebook wav2vec2-base-960h [model](https://huggingface.co/facebook/wav2vec2-base-960h) (Torch/TF)
            - Facebook wav2vec2-large-960h [model](https://huggingface.co/facebook/wav2vec2-large-960h) (Torch)
            - Vasudevgupta tf-wav2vec2-base [model](https://huggingface.co/vasudevgupta/tf-wav2vec2-base) (TF)
            - Vasudevgupta gsoc-wav2vec2 [model](https://huggingface.co/vasudevgupta/gsoc-wav2vec2) (TF)
            - Vasudevgupta gsoc-wav2vec2-960h [model](https://huggingface.co/vasudevgupta/gsoc-wav2vec2-960h) (TF)