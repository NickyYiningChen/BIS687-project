import torch
from torch import nn
from torch.autograd import *
import torch.nn.functional as F
from gen_embed import generate_embeddings
import sys



class CustomLSTM(nn.Module):
    def __init__(self, args):
        super(CustomLSTM, self).__init__()
        self.args = args
        self.relu = nn.ReLU ( )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.lstm_size = args.lstm_size 
        self.output_size = 1

        self.main_lstm = nn.LSTM (input_size=args.embed_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True)


        # unstructure
        self.vocab_embedding = nn.Embedding (args.vocab_size, args.embed_size )
        self.vocab_mapping = nn.Sequential(
                nn.Linear(args.embed_size * 2, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                nn.Linear(args.embed_size, args.embed_size),
                )


        self.value_embedding = nn.Embedding.from_pretrained(generate_embeddings(args.embed_size, args.split_num + 1))
        self.value_mapping = nn.Sequential(
                nn.Linear (args.embed_size * 2, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                )
        
        self.demo_embedding = nn.Embedding (args.n_ehr, args.embed_size )
        self.demo_processing = nn.Sequential(
            nn.Linear(args.embed_size*2, args.hidden_size),  # First dense layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, args.embed_size),  # Second dense layer (optional)
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.final_map = nn.Sequential (
            nn.Linear (self.lstm_size*2, self.lstm_size),
            nn.ReLU ( ),
            nn.Linear (self.lstm_size, self.lstm_size),
            nn.ReLU ( )
        )
        self.final_out = nn.Sequential (
            nn.Linear (self.lstm_size * 2, self.lstm_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
            nn.Linear (self.lstm_size, self.output_size),
        )

        self.combiner = nn.Sequential (
                nn.Linear (self.lstm_size * 3, self.lstm_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                nn.Linear (self.lstm_size, self.output_size),
                )

    def embedd_value(self, x):
        indices, values = x
        embedded_indices = self.vocab_embedding(indices.view(-1))  # Flatten the indices before embedding
        embedded_values = self.value_embedding(values.view(-1))    # Flatten the values before embedding
        concatenated_embeddings = torch.cat((embedded_indices, embedded_values), dim=1)
        mapped_embeddings = self.value_mapping(concatenated_embeddings)
        final_embeddings = mapped_embeddings.view(*indices.size(), -1)  # Preserve the original batch and sequence dimensions

        return final_embeddings

    
    def process_demographic_data(self, demo):
        demographic_data = demo.view(-1)
        embedded_demographics = self.demo_embedding(demographic_data).view(*demo.size(), -1)
        lstm_demo_out, _ = self.main_lstm(embedded_demographics)
        transposed_demo_out = lstm_demo_out.transpose(1, 2)
        pooled_demo_out = self.pooling(transposed_demo_out)
        processed_demo_out = self.demo_processing(pooled_demo_out.view(pooled_demo_out.size(0), -1))
        return processed_demo_out

    def process_main_lstm(self, x):
        lstm_out_x, _ = self.main_lstm(x)
        mapped_x = self.final_map(lstm_out_x)
        transposed_x = mapped_x.transpose(1, 2)
        pooled_x_final = self.pooling(transposed_x)
        final_x = pooled_x_final.view(pooled_x_final.size(0), -1)
        return final_x

    def process_notes(self, content):
        lstm_content_out, _ = self.main_lstm(content)
        mapped_content = self.vocab_mapping(lstm_content_out)
        transposed_content = mapped_content.transpose(1, 2)
        pooled_content = self.pooling(transposed_content)
        final_content = pooled_content.view(pooled_content.size(0), -1)
        return final_content

    def forward(self, x, t, demo, notes=None):
        embedded_x = self.embedd_value(x)
        pooled_x = self.visit_pooling(embedded_x)

        processed_d = self.process_demographic_data(demo)
        final_x = self.process_main_lstm(pooled_x)

        if notes is not None:
            final_content = self.process_notes(notes)
        else:
            final_content = torch.zeros_like(final_x)  # Handle missing content by zero padding

        
        #Concatenates the given sequence of seq tensors in the given dimension
        combined_output = torch.cat((final_x, final_content, processed_d), dim=1)
        output = self.combiner(combined_output)
        return output
    
    def visit_pooling(self, tensor):

        tensor_size = tensor.size()
        merged_tensor = tensor.view(tensor_size[0] * tensor_size[1], tensor_size[2], tensor.size(3))
        transposed_tensor = merged_tensor.transpose(1, 2).contiguous()
        pooled_tensor = self.pooling(transposed_tensor)
        reshaped_tensor = pooled_tensor.view(tensor_size[0], tensor_size[1], tensor_size[3])

        return reshaped_tensor
    
