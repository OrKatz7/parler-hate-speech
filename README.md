# parler-hate-speech
## Install
```
git clone https://github.com/OrKatz7/parler-hate-speech
cd parler-hate-speech
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
cd ..
pip install -U easynmt
pip install thai-segmenter
pip install -U protobuf==3.20.0
pip install -U iterative-stratification==0.1.7
pip install -U transformers==4.21.2
pip install -U tokenizers==0.12.1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```
## How To Run
```
./run_train.sh
```
