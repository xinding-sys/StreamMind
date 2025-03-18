from transformers import MistralConfig, MistralForCausalLM

import json


mis_config = MistralConfig()
mis_config.vocab_size =2
mis_config.num_hidden_layers =4

model = MistralForCausalLM(config=mis_config)

