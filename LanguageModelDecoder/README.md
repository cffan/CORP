# Introduction
This is the repo for langauage model decoder. Codes are based on [WeNet](https://github.com/wenet-e2e/wenet) and [Kaldi](https://github.com/kaldi-asr/kaldi).

# Dependencies
```
CMake >= 3.14
gcc >= 10.1
pytorch==1.13.1
```

Please note that this library uses libtorch 1.13.1. If you have other versions of pytorch installed, you may need to uninstall them first and then install the correct version.


# Instructions for build and run the language model decoder

Step 1: install the lanauge model decoder first:
```bash
cd runtime/server/x86
python setup.py install
```

Step 2: download the [language model]().

Step 3: test if you can run the following code. You need ~40GB of memory to run this code.
```python
import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils

lm_decoder = lmDecoderUtils.build_lm_decoder(
    'path/to/your/language/model',
)
```



