**Nikolay Vorontsov** 
--- 
**LLMs and GenAI for NLP, 2024**  
<mark>Report on the Exercises in Labs 1 – 6</mark>  
GitHub repository: **[nicksnlp](https://github.com/nicksnlp/LLM-course-2024_Nick)**   

This is what I have done: 

## Week4

### Fine-tuning a model with LLMs, PEFT, LoRA  
**supervised_finetuning.ipynb**

I've created accounts on HuggingFace, Weights&Biases, I will use my regualar Google account. I use access tokens in *Secrets* on Colab.

I am running notebook on Colab Pay-As-You-Go, T4 GPU.

- [X] Loaded dataset, here is an example:  
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Give three tips for staying healthy

### Response:
1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.

2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.

3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.
```
- [x] Created config object
- [x] Downloaded the model
- [x] Downloaded tokeniser

Applied for an Academic account at **W&B**. Now I can hopefully visualise and save models easier.

I have tried to set-up training parametres, but keep getting errors for different arguments (*max_seq_length*, *dataset_text_field*, *packing*) of their incompatibility with SFTTrainer:

For example:
```
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'max_seq_length'
```
#### Solution:
I have changed `tokenizer` into `proccessing_class` in `SDTTrainer`.

Commenting out `max_seq_length=None` etc. from the `SFTTTrainer` arguments seems also to work.

For now I will follow the pre-existed setup, with no truncation, padding, max_length. But there is an option to add the following into the code:

```
# Define the maximum sequence length (optional)
max_length = 512  # Set a reasonable length for your model

# Function to process the dataset by tokenizing and padding/truncating
def tokenize_function(batch):
    # Tokenize the 'text' field
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=max_length)

# Apply the function to the entire dataset
dataset = dataset.map(tokenize_function, batched=True)
```
Since I have got this warning:
```
/usr/local/lib/python3.10/dist-packages/trl/trainer/sft_trainer.py:300: UserWarning: You passed a processing_class with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to overflow issues when training a model in half-precision. You might consider adding `processing_class.padding_side = 'right'` to your code.
  warnings.warn(
```
I have added the following line into the code, before defining the `trainer`:
```
tokenizer.padding_side = "right"
```
However, I am not completely sure now if `right` was the correct option for padding, or whether I could get away with no padding.  
Here they used the `right`:
- [Fine-Tuning Mistral](https://wandb.ai/byyoung3/ml-news/reports/Fine-Tuning-Mistral-7B-on-Python-Code-With-A-Single-GPU---Vmlldzo1NTg0NzY5)  
  
While for generation the `left` padding side is suggested:
- [Generation with LLMs](https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side)

**A note:**   
If I was to run training several times, I should consider adding specific names (  `name="small_run_1K"` ) for training runs for better management in W&B into `wand.init(...)`, as well as:
```
training_arguments = TrainingArguments(
    output_dir="./results",
    run_name="unique_run_name",  # Add a custom name here
...
```

- [x] Send `trainer.train()` to run...  
    Estimated time needed for training (1 epoch): \~ 8 hours  

    UPDATE: Unfortunately, I have been cut off from colab, after it was almost done.  
    [561/625 7:03:31 < 48:29, 0.02 it/s, Epoch 0.90/1]  
    
    ---

    **Verdict**: Running this notebook with the available resources, without saving checkpoints outside of Colab was not a good idea... The data is lost and the time too...  
    
    I will change the subset into 1K to check the pipeline, and retrain, I will call the model *shrimp*

    Also I will mount the Google Drive and to save checkpoints and other data there, so I could use checkpoints to resume training if it fails during the process. If Colab fails, the environments and all the data gets cleared too.  

    For that reason I have added `resume_from_checkpoint=True` and `save_total_limit=3` into `TrainingArguments`. For 1K, there should be 63 steps, so I have set up `save_steps=10`, this can be 50 for 10K datapoints (625 steps) training. I have first tested the pipeline with 100 datapoints, and then run it with 1K.
    
    **Possible alternative 1**: Save the checkpoints and models to W&B, it then needs to be loaded for resuming, with a callback function as an *artifact*...  
    **Possible alternative 2**: Do the whole training somewhere outside of Colab with a SLURM script.

    ---
- [x] Evaluate training results and loss with W&B  
    
    For the failed **10K** run went pretty well with the loss function looking as follows.

    <img src="./pics/W&B%20Chart%2030_12_2024,%2000_58_08.svg" alt="loss 10K" width="700" />

    The training on **1K datapoints** the loss gained **1.6542** at step 60. This must be lower than in 10K since the warm-up was shorter.  

    However, one need to decide what metrics/parameters to use to properly evaluate the model... This stays beyond the scope of this exercise, we somehow evaluate the results with the `stream` function, indeed while in 100 datapoints test-run the results very rather hallucinative, with 1K, although with a lot of repetitive information they are already reasonably good, but what is good depends of course on our needs...

- [x] Save the model (Where!? Yes, in Colab environment...)  

    Saving with the name `new_model` caused issues when later pushing the model, therefore I have saved it with a different path, not `new_model`.

- [x] Loaded the base model
    When loading `base_model` I have set up `device_map = {"": 0}`, and implemented quantisation, by adding: `quantization_config=bnb_config` into parameters. `bnb_config` was defined earlier.

- [x] Merged the `base_model` and `new_model` and pushed into HuggingFace.
   
     The new model has 3.87B parameters.

- [X] Created a model card for this model: https://huggingface.co/nicksnlp/shrimp/blob/main/README.md  

Not hasslefree, but everything worked at the end. The main issue was in adapting the code so it correctly processes paths when saving to Hugging Face, adding functionality to keep intermediate models and resume training from checkpoints, as well as adding correctly quantisation and padding. I've learned a lot of things about setting things up, and the process!

Selecting another base model and dataset
---
I wanted to try using T5 Google Model and a labelled QA dataset to adapt it for detecting hallucinations in texts. Unfortunately, even *flan-t5-small* has 77M parameters, so I will save this idea for the future. Ideally, I will try to use W&B and run the training on Puhti with a SLURM script.

An alternative is to use one of the llama family model, they range from 1B to 70B in size. For example `meta-llama/Llama-2-7b-hf`. Finding a good dataset is problematic, I will need to adapt an existing one. For now I will use a sample dictionary. The code is implemented in the notebook *Fine-tune_llama.ipynb*.


Utilising DPO instead of supervised fine-tuning
---
**Fine_tune_a_Mistral_7b_model_with_DPO.ipynb**

Since this is a bonus exercise, I will hopefully do it later on...

### References:
1. https://chatgpt.com/share/677021c2-0128-800b-957b-511b29768fd4
2. https://chatgpt.com/share/67708817-1a4c-800b-a17a-c99bcdcbd05d
3. https://chatgpt.com/share/67709535-13c0-800b-b07a-2446d28e701a
4. https://chatgpt.com/share/67714f3a-390c-800b-8a6c-2da3d5c5815b
5. https://chatgpt.com/share/6771c958-592c-800b-a846-1a10425d06f0
6. https://chatgpt.com/share/67729fee-da9c-800b-808a-28a722cd3174

---