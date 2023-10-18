from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5Config
from transformers.generation_utils import GenerationMixin
from constrained_module.constrained_generation import ConstrainedGenerationMixin


class T5ForSequenceClassification(T5ForConditionalGeneration):
    """
    Modeling hierarchical text classification as Seq2Seq model using T5 as backbone.
    Generation is done via beam search with or without constraining with respect to hierarchy.
    """

    def __init__(self, model='t5-base', target_type="dfs", max_target_length=20, constrained=True):
        config = T5Config.from_pretrained(model)
        super().__init__(config)
        self.target_type = target_type
        self.task_prefix = "Classify:"
        self.max_source_length = 512
        self.max_target_length = max_target_length
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.config = self.model.config
        self.dynamic_vocab_cache = {}
        self.constrained = constrained

    def forward(self, text, labels):
        encoding = self.tokenizer(
            [self.task_prefix + sequence for sequence in text],
            padding="longest",
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        target_encoding = self.tokenizer(
            labels,
            padding="longest",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        labels = target_encoding.input_ids
        if not self.constrained:
            labels[labels == self.tokenizer.pad_token_id] = -100
            return labels, self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

        return labels, self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

    def generate(
            self,
            inputs, next_token_mapping=None, set_all_labels=None,
            **kwargs
    ):
        encoding = self.tokenizer(
            [self.task_prefix + sequence for sequence in inputs],
            padding="longest",
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        if not self.constrained:
            return GenerationMixin.generate(self.model,
                                            input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            num_beams=3, max_length=self.max_target_length,
                                            top_k=10, top_p=0.95, early_stopping=False,
                                            **kwargs)
        return ConstrainedGenerationMixin.generate(self.model, self.dynamic_vocab_cache, target_type=self.target_type,
                                                   input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   tokenizer=self.tokenizer, num_beams=3,
                                                   max_length=self.max_target_length,
                                                   next_token_mapping=next_token_mapping, set_all_labels=set_all_labels,
                                                   top_k=10, top_p=0.95, early_stopping=False,
                                                   **kwargs)
