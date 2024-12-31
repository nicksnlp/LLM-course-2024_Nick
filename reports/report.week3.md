
## Week 3

The assignment was to find the ways to evaluate models, and more specifically their ability to produce adequate, similar in terms of content results but within different domains.

**Zero-shot prompting**  
I had different ideas about how this assignment can be approached. And I have come to a decision that a proper experiment requires the following set-up:  
1. Selection of models (e.g. from Hugging Face), smaller would be more interesting for evaluation purposes (the larger seem to be harder to break);
2. a large selection of questions, within the specified domains, a dataset (this can be an augmented dataset, produced by an llm of a choice, or some of already available QA datasets);
3. a system that will loop through the dataset and deliver prompts into models, and register the outputs;
4. a system that compares the outputs from the models against the gold-standard, or a reference set, in terms of adequacy. An adaptation of ROUGE-metric may be an option to use here, or a specifically-designed system for that purpose, that will check the output irrespective of their style/language, based on the defined parameters, this can be also be an llm-system.

*Unfortunately, I have not run the experiment itself yet, which would be a nice thing to do in the future.*

**Few-shot prompting**  
I have also tried prompting a model (GPT-4o) in an dialogue set-up, by asking it for general financial advices in a informal/slang type of language in Russian. As it proved, the model was very good in adapting to a specific language-style while still giving a reasonable advice. I suppose the dialogue systems, like chat-gpt, have some solid external architecture that adjusts the output before delivering, the model keeps remembering the previous dialogue, and is well designed for a conversation. The only way to shift its output was to trigger a different topic, which may seem more important from some point of view.

For some more specific tasks, like checking the code, the output of chat-gpt can much more often seem inadequate in terms of the questions asked, and a more accurate line up of question is important.

I have also did an interesting test and asked the model to *"give me a code to remove the outdated food from my fridge"*, which model handled relatively well. I suppose asking models ridiculous or ambiguous questions may also be a way to check their adequacy.

Unfortunately, I did not keep good references for those prompts.

---