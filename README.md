<img src="./docs/source/_static/images/nnsight_logo.svg" alt="drawing" style="width:200px;float:left"/>

# nnsight 
![PyPI - Version](https://img.shields.io/pypi/v/nnsight)

[nnsight.net](www.nnsight.net)

The `nnsight`  package enables interpreting and manipulating the internals of deep learned models.

#### Installation

Install this package through pip by running:

`pip install nnsight`

#### Examples

Here is a simple example where we run the nnsight API locally on gpt2 and save the hidden states of the last layer:

```python
from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cuda')

with model.generate(max_new_tokens=1) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states = model.transformer.h[-1].output[0].save()

output = generator.output
hidden_states = hidden_states.value
```

Lets go over this piece by piece.

We import the `Model` object from the `nnsight` module and create a gpt2 model using the huggingface repo ID for gpt2, `'gpt2'`. This accepts arguments to create the model including `device_map` to specify which device to run on.

```python
from nnsight import LanguageModel

model = LanguageModel('gpt2',device_map='cuda')
```

Then, we create a generation context block by calling `.generate(...)` on the model object. This denotes we wish to actually generate tokens given some prompts.

Keyword arguments are passed downstream to [AutoModelForCausalLM.generate(...)](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate). Refer to the linked docs for reference.


```python
with model.generate(max_new_tokens=3) as generator:
```

Now calling `.generate(...)` does not actually initialize or run the model. Only after the `with generator` block is exited, is the acually model loaded and ran. All operations in the block are "proxies" which essentially creates a graph of operations we wish to carry out later.


Within the generation context, we create invocation contexts to specify the actual prompts we want to run:


```python
with generator.invoke('The Eiffel Tower is in the city of') as invoker:
```

Within this context, all operations/interventions will be applied to the processing of this prompt.

```python
hidden_states = model.transformer.h[-1].output[0].save()
```

On this line were saying, access the last layer of the transformer `model.transformer.h[-1]`, access its output `.output`, index it at 0 `.output[0]`, and save it `.save()`

A few things, we can see the module tree of the model by printing the model. This allows us to know what attributes to access to get to the module we need.
Running `print(model)` results in:

```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```

`.output` returns a proxy for the output of this module. This essentially means were saying, when we get to the output of this module during inference, grab it and perform any operations we define on it (which also become proxies). There are two operational proxies here, one for getting the 0th index of the output, and one for saving the output. We take the 0th index because the output of gpt2 transformer layers are a tuple where the first index are the actual hidden states (last two indicies are from attention). We can call `.shape` on any proxies to get what shape the value will eventually be. 
Running `print(model.transformer.h[-1].output.shape)` returns `(torch.Size([1, 10, 768]), (torch.Size([1, 12, 10, 64]), torch.Size([1, 12, 10, 64])))`

During processing of the intervention computational graph we are building, when the value of a proxy is no longer ever needed, its value is dereferenced and destroyed. However calling `.save()` on the proxy informs the computation graph to save the value of this proxy and never destroy it, allowing us to access to value after generation.

After exiting the generator context, the model is ran with the specified arguments and intervention graph. `generator.output` is populated with the actual output and `hidden_states.value` will contain the value.

```python
output = generator.output
hidden_states = hidden_states.value

print(output)
print(hidden_states)
```

returns:

```
tensor([[ 464,  412,  733,  417, 8765,  318,  287,  262, 1748,  286, 6342]],
       device='cuda:0')
tensor([[[ 0.0505, -0.1728, -0.1690,  ..., -1.0096,  0.1280, -1.0687],
         [ 8.7494,  2.9057,  5.3024,  ..., -8.0418,  1.2964, -2.8677],
         [ 0.2960,  4.6686, -3.6642,  ...,  0.2391, -2.6064,  3.2263],
         ...,
         [ 2.1537,  6.8917,  3.8651,  ...,  0.0588, -1.9866,  5.9188],
         [-0.4460,  7.4285, -9.3065,  ...,  2.0528, -2.7946,  0.5556],
         [ 6.6286,  1.7258,  4.7969,  ...,  7.6714,  3.0682,  2.0481]]],
       device='cuda:0')
```



---

###### Operations

Most* basic operations and torch operations work on proxies and are added to the computation graph. 

```python
from nnsight import LanguageModel
import torch 

model = LanguageModel('gpt2', device_map='cuda')

with model.generate(max_new_tokens=1) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states_pre = model.transformer.h[-1].output[0].save()

        hs_sum = torch.sum(hidden_states_pre).save()

        hs_edited = hidden_states_pre + hs_sum

        hs_edited = hs_edited.save()

print(hidden_states_pre.value)
print(hs_sum.value)
print(hs_edited.value)
```

In this example we get the sum of the hidden states and add them to the hidden_states themselves (for whatever reason). By saving the various steps, we can see how the values change.

```
tensor([[[ 0.0505, -0.1728, -0.1690,  ..., -1.0096,  0.1280, -1.0687],
         [ 8.7494,  2.9057,  5.3024,  ..., -8.0418,  1.2964, -2.8677],
         [ 0.2960,  4.6686, -3.6642,  ...,  0.2391, -2.6064,  3.2263],
         ...,
         [ 2.1537,  6.8917,  3.8651,  ...,  0.0588, -1.9866,  5.9188],
         [-0.4460,  7.4285, -9.3065,  ...,  2.0528, -2.7946,  0.5556],
         [ 6.6286,  1.7258,  4.7969,  ...,  7.6714,  3.0682,  2.0481]]],
       device='cuda:0')
tensor(501.2957, device='cuda:0')
tensor([[[501.3461, 501.1229, 501.1267,  ..., 500.2860, 501.4237, 500.2270],
         [510.0451, 504.2014, 506.5981,  ..., 493.2538, 502.5920, 498.4279],
         [501.5916, 505.9643, 497.6315,  ..., 501.5348, 498.6892, 504.5219],
         ...,
         [503.4493, 508.1874, 505.1607,  ..., 501.3545, 499.3091, 507.2145],
         [500.8496, 508.7242, 491.9892,  ..., 503.3485, 498.5010, 501.8512],
         [507.9242, 503.0215, 506.0926,  ..., 508.9671, 504.3639, 503.3438]]],
       device='cuda:0')
       
```

---
###### Setting

We often not only want to see whats happening during computation, but intervene and edit the flow of information. 

```python
from nnsight import LanguageModel
import torch 

model = LanguageModel('gpt2', device_map='cuda')

with model.generate(max_new_tokens=1) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states_pre = model.transformer.h[-1].output[0].save()

        noise = (0.001**0.5)*torch.randn(hidden_states_pre.shape)

        model.transformer.h[-1].output[0] = hidden_states_pre + noise

        hidden_states_post = model.transformer.h[-1].output[0].save()

print(hidden_states_pre.value)
print(hidden_states_post.value)
```
In this example, we create a tensor of noise to add to the hidden states. We then add it, use the assigment `=` operator to update the tensors of `.output[0]` with these new noised values. 

We can see the change in the results:

```
tensor([[[ 0.0505, -0.1728, -0.1690,  ..., -1.0096,  0.1280, -1.0687],
         [ 8.7494,  2.9057,  5.3024,  ..., -8.0418,  1.2964, -2.8677],
         [ 0.2960,  4.6686, -3.6642,  ...,  0.2391, -2.6064,  3.2263],
         ...,
         [ 2.1537,  6.8917,  3.8651,  ...,  0.0588, -1.9866,  5.9188],
         [-0.4460,  7.4285, -9.3065,  ...,  2.0528, -2.7946,  0.5556],
         [ 6.6286,  1.7258,  4.7969,  ...,  7.6714,  3.0682,  2.0481]]],
       device='cuda:0')
tensor([[[ 0.0674, -0.1741, -0.1771,  ..., -0.9811,  0.1972, -1.0645],
         [ 8.7080,  2.9067,  5.2924,  ..., -8.0253,  1.2729, -2.8419],
         [ 0.2611,  4.6911, -3.6434,  ...,  0.2295, -2.6007,  3.2635],
         ...,
         [ 2.1859,  6.9242,  3.8666,  ...,  0.0556, -2.0282,  5.8863],
         [-0.4568,  7.4101, -9.3698,  ...,  2.0630, -2.7971,  0.5522],
         [ 6.6764,  1.7416,  4.8027,  ...,  7.6507,  3.0754,  2.0218]]],
       device='cuda:0')
```

Note: Only assigment updates of tensors works with this functionality. 

---
###### Multiple Token Generation

When generating more than one token, use `invoker.next()` to denote following interventions should be applied to the subsequent generations.

Here we again generate using gpt2, but generate three tokens and save the hidden states of the last layer for each one:

```python
from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cuda')

with model.generate(max_new_tokens=3) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states1 = model.transformer.h[-1].output[0].save()

        invoker.next()
        
        hidden_states2 = model.transformer.h[-1].output[0].save()

        invoker.next()
        
        hidden_states3 = model.transformer.h[-1].output[0].save()


output = generator.output
hidden_states1 = hidden_states1.value
hidden_states2 = hidden_states2.value
hidden_states3 = hidden_states3.value
```
---

###### Token Based Indexing


When indexing hidden states for specific tokens, use `.token[<idx>]` or `.t[<idx>]`.
This is because if there are multiple invocations, padding is performed on the left side so these helper functions index from the back.

Here we just get the hidden states of the first token:

```python
from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cuda')

with model.generate(max_new_tokens=1) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states = model.transformer.h[-1].output[0].t[0].save()

output = generator.output
hidden_states = hidden_states.value
```

---

###### Cross Prompt Intervention


Intervention operations work cross prompt! Use two invocations within the same generation block and operations can work between them.

In this case, we grab the token embeddings coming from the first prompt, `"Madison square garden is located in the city of New"` and replace the embeddings of the second prompt with them.

```python
from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cuda')

with model.generate(max_new_tokens=3) as generator:
    
    with generator.invoke("Madison square garden is located in the city of New") as invoker:

        embeddings = model.transformer.wte.output

    with generator.invoke("_ _ _ _ _ _ _ _ _ _") as invoker:

        model.transformer.wte.output = embeddings

print(model.tokenizer.decode(generator.output[0]))
print(model.tokenizer.decode(generator.output[1]))
```

This results in:

```
Madison square garden is located in the city of New York City.
_ _ _ _ _ _ _ _ _ _ York City.
```

We also could have entered a pre-saved embedding tensor as shown here:

```python
from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cuda')

with model.generate(max_new_tokens=3) as generator:
    
    with generator.invoke("Madison square garden is located in the city of New") as invoker:

        embeddings = model.transformer.wte.output.save()

print(model.tokenizer.decode(generator.output[0]))
print(embeddings.value)

with model.generate(max_new_tokens=3) as generator:

    with generator.invoke("_ _ _ _ _ _ _ _ _ _") as invoker:

        model.transformer.wte.output = embeddings.value

print(model.tokenizer.decode(generator.output[0]))

```
---

###### Ad-hoc Module

Another thing we can do is apply modules in the model's module tree at any point during computation, even if it's out of order.

```python
from nnsight import LanguageModel
import torch

model = LanguageModel("gpt2", device_map='cuda')

with model.generate() as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:
        
        hidden_states = model.transformer.h[-1].output[0]
        hidden_states = model.lm_head(model.transformer.ln_f(hidden_states)).save()
        tokens = torch.softmax(hidden_states, dim=2).argmax(dim=2).save()
        
print(hidden_states.value)
print(tokens.value)
print(model.tokenizer.decode(tokens.value[0]))

```

Here we get the hidden states of the last layer like usual. We also chain apply `model.transformer.ln_f` and `model.lm_head` in order to "decode" the hidden states into vocabularly space.
Applying softmax and then argmax allows us to then transform the vocabulary space hidden states into actually tokens which we can then use the tokenizer to decode.

The output looks like:

```
tensor([[[ -36.2874,  -35.0114,  -38.0793,  ...,  -40.5163,  -41.3759,
           -34.9193],
         [ -68.8886,  -70.1562,  -71.8408,  ...,  -80.4195,  -78.2552,
           -71.1206],
         [ -82.2950,  -81.6519,  -83.9941,  ...,  -94.4878,  -94.5194,
           -85.6998],
         ...,
         [-113.8675, -111.8628, -113.6634,  ..., -116.7652, -114.8267,
          -112.3621],
         [ -81.8531,  -83.3006,  -91.8192,  ...,  -92.9943,  -89.8382,
           -85.6898],
         [-103.9307, -102.5054, -105.1563,  ..., -109.3099, -110.4195,
          -103.1395]]], device='cuda:0')
tensor([[ 198,   12,  417, 8765,  318,  257,  262, 3504, 7372, 6342]],
       device='cuda:0')

-el Tower is a the middle centre Paris
```

---

###### Running Remotely


Running the nnsight API remotely on LLaMA 65b and saving the hidden states of the last layer:

```python
from nnsight import LanguageModel

model = LanguageModel('decapoda-research/llama-65b-hf')
with model.generate(server=True, max_new_tokens=1) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states = model.model.layers[-1].output[0].save()

output = generator.output
hidden_states = hidden_states.value
```

More examples can be found in `nnsight/examples/` and at [nnsight.net](www.nnsight.net)

### Citation

If you use `nnsight` in your research, please cite using the following

```bibtex
@software{nnsight,
author = {Jaden Fiotto-Kaufman},
license = {MIT},
title = {{nnsight: The package for interpreting and manipulating the internals of deep learned models.
}},
url = {https://github.com/JadenFiotto-Kaufman/nnsight}
}
``````