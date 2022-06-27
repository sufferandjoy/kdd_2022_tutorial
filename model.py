import torch
import torch.nn as nn
from transformers import BertModel


class TeacherModel(nn.Module):
    def __init__(self, args):
        super(TeacherModel, self).__init__()
        self.model = BertModel.from_pretrained(args.teacher_pretrained)
        hidden_size = int(args.teacher_pretrained.split('/')[1].split('_')[3].split('-')[-1])
        self.ff = nn.Linear(hidden_size, 1)

    def forward(self, ids, mask, token_type_ids):
        bert_output = self.model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = torch.sigmoid(self.ff(bert_output.pooler_output))
        return output


class TwinBERT(nn.Module):
    def __init__(self, args):
        super(TwinBERT, self).__init__()
        self.encoder_model = BertModel.from_pretrained(args.student_pretrained)

    def forward(self, seq1, mask1, seq2, mask2):
        output_1 = self.encoder_model(seq1, attention_mask=mask1).pooler_output
        output_2 = self.encoder_model(seq2, attention_mask=mask2).pooler_output
        cosine_similarity = nn.functional.cosine_similarity(output_1, output_2).unsqueeze(-1)
        return cosine_similarity