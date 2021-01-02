"""
https://github.com/minimaxir/gpt-2-simple
https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce#scrollTo=aeXshJM-Cuaf
https://minimaxir.com/2019/09/howto-gpt2/
"""

import gpt_2_simple as gpt2
import os
import requests

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "shakespeare.txt"
if not os.path.isfile(file_name):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data = requests.get(url)
    
    with open(file_name, 'w') as f:
        f.write(data.text)

sess = gpt2.start_tf_sess()
#gpt2.finetune(sess,
#              file_name,
#              model_name=model_name,
#              steps=1000)   # steps is max number of training steps

gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=1000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=200,
              save_every=500
              )

gpt2.generate(sess)