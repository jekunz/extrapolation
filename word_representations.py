import torch
from transformers import BertTokenizer, BertModel, BertConfig

"""

class Bert

access via get_bert()
input: sentence represented as a string
returns: list with torch states 

"""


class Bert:
    def __init__(self, layer):
        self.config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_basic_tokenize=True)
        self.bert_layer = layer
        self.model.eval()

    # align word pieces with words
    @staticmethod
    def collect_pieces(tokenized_text):
        output = []
        curr_token = []
        seq_length = len(tokenized_text)

        for i in range(seq_length):
            curr_piece = tokenized_text[i]
            curr_token.append((i, curr_piece))

            if i < seq_length - 1:
                next_piece = tokenized_text[i + 1]
                if not next_piece.startswith('##'):
                    output.append(curr_token)
                    curr_token = []

        output.append(curr_token)
        return output

    def get_bert(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            target_layer = outputs[2][self.bert_layer]

        collected_pieces = Bert.collect_pieces(tokenized_text)
        token_states = []
        for t in collected_pieces:
            token_index = t[-1][0]  # taking last word piece
            token_states.append(target_layer[0, token_index])
        return token_states[1:len(token_states)]    # -1 ???
