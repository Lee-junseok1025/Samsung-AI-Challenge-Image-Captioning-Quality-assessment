import torch
from MANIQA import *


class quality_scoring(nn.Module):
    def __init__(self, embed_dim, num_outputs=1,drop=0.1):
        super(quality_scoring, self).__init__()

        self.fc_score = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(embed_dim // 2, num_outputs),
                    nn.ReLU()
                )
        self.fc_weight = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(embed_dim // 2, num_outputs),
                nn.Sigmoid()
        )
    def forward(self,x):
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MANIQA2transformer(nn.Module):
    def __init__(self, vocab_size, feature_dim=384, embed_dim=512, d_model=512):
        super(MANIQA2transformer, self).__init__()
        self.d_model = d_model
        # Image feature extraction using MANIQA
        self.cnn = MANIQA(embed_dim=768, num_outputs=384, dim_mlp=768,
        patch_size=8, img_size=224, window_size=4,
        depths=[2,2], num_heads=[4,4], num_tab=2, scale=0.8)
        # Remove the last fully connected layer to get features
        self.vocab_size = vocab_size
        # Image quality assessment head
        self.scoring = quality_scoring(embed_dim=384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.3)
        self.selu = nn.SELU()
        # Captioning head
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=(d_model+feature_dim),nhead=8)
        self.transformer_e = nn.TransformerEncoder(encoder_layers,num_layers=8)
        self.mapping_decoder = nn.Linear((d_model+feature_dim),d_model)
        self.mapping_norm = nn.LayerNorm(d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model=d_model,nhead=8)
        self.transformer_d = nn.TransformerDecoder(decoder_layers,num_layers=8)
        
        # decoder_layers = nn.TransformerDecoderLayer(d_model=896,nhead=4)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, images, src=None, tgt=None, tgt_mask=None):
        # CNN
        features = self.cnn(images)
        features = features.transpose(-2,-1)
        features = self.avgpool(features)
        features_flat = features.squeeze(2)


        # Image quality regression
        mos = self.scoring(features_flat)
        if src is not None:
            src = self.dropout_2(self.embedding(src)) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            src = src.permute(1,0,2)
            tgt = self.dropout_2(self.embedding(tgt)) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt)
            tgt = tgt.permute(1,0,2)
            combined_e = torch.cat((src, features.permute(2,0,1).repeat(src.size(0), 1, 1)), dim=2)
            encoder_out = self.transformer_e(combined_e)
            encoder_out = self.mapping_norm(self.mapping_decoder(encoder_out))
            decoder_out = self.transformer_d(tgt,encoder_out, tgt_mask)
            decoder_out = decoder_out.permute(1,0,2)
            output = self.fc(decoder_out.squeeze(1))
           
            
            return mos, output
        else:
          return mos, None
    def generate_square_subsequent_mask(self, sz: int):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int):
        return (matrix == pad_token)

    def inference(self, features, src=None, tgt=None):
        features = features.transpose(-2,-1)
        features = self.avgpool(features)

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.permute(1,0,2)

        tgt_mask = self.get_tgt_mask(tgt.size(1)).cuda()
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        tgt = tgt.permute(1,0,2)

        # src torch.Size([4, 1, 512])
        # features torch.Size([4, 384, 1])
        combined_e = torch.cat((src, features.permute(0,2,1).repeat(src.size(0), 1, 1)), dim=2)
        combined_d = torch.cat((tgt, features.permute(0,2,1).repeat(src.size(0), 1, 1)), dim=2)
            
        encoder_out = self.transformer_e(combined_e)
        decoder_out = self.transformer_d(combined_d, encoder_out, tgt_mask)
        decoder_out = decoder_out.permute(1,0,2)
        output = self.fc(decoder_out.squeeze(1))
        return output
    def sample_beam_search(self, features, beam_size):
        k = beam_size
        vocab_size = self.vocab_size
        features = features.transpose(-2,-1)
        features = self.avgpool(features)
        encoder_size = features.size(1)
        features = features.view(1, 1, encoder_size)
        inputs = features.expand(k, 1, encoder_size)

        top_k_scores = torch.zeros(k, 1).cuda()
        seqs = torch.zeros(k, 1).long().cuda()
        complete_seqs = list()
        complete_seqs_scores = list()
        input_src = torch.tensor([word2idx['<SOS>'] for i in range (k)]).view(1,-1).cuda()
        input_tgt = torch.tensor([word2idx['<PAD>'] for i in range (k)]).view(1,-1).cuda()
        step = 1

        while True:
            if step == 1:
                outputs = self.inference(features,input_src,input_tgt)
            else:
                outputs = self.inference(features,src,tgt)

            scores = F.log_softmax(outputs.squeeze(0), dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)  # (s)

            
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            # 새로둔 단어를 seqs에 더한다.
            if step==1:
                seqs = next_word_inds.unsqueeze(1)
            else:
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word2idx['<EOS>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            # cell = cell[:, prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            src = k_prev_words.view(1,-1)
            tgt = torch.tensor([word2idx['<PAD>'] for i in range (k)]).view(1,-1).cuda()
            if step > k:
                break
            step += 1
        return seqs[0]
    def generate_caption(self,features,src,beam_size,max_len=50):
        # Inference part
        # Given the image features generate the captions
        
        batch_size = features.size(0)
        features = features.transpose(-2,-1)
        features = self.avgpool(features)
        captions = [i.item() for i in src]
        #starting input
        src_token = torch.tensor(src[-1]).view(1,-1).cuda()  
        tgt_token = torch.tensor(word2idx['<PAD>']).view(1,-1).cuda()
        for i in range(max_len-(beam_size+1)):
            src = self.embedding(src_token) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            src = src.permute(1,0,2)

            tgt = self.embedding(tgt_token) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt)
            tgt = tgt.permute(1,0,2)
            tgt_mask = self.get_tgt_mask(1).cuda()
            
            combined_e = torch.cat((src, features.permute(2,0,1).repeat(src.size(0), 1, 1)), dim=2)
            combined_d = torch.cat((tgt, features.permute(2,0,1).repeat(tgt.size(0), 1, 1)), dim=2)
            
            encoder_out = self.transformer_e(combined_e)
            decoder_out = self.transformer_d(combined_d, encoder_out, tgt_mask)
            decoder_out = decoder_out.permute(1,0,2)
            output = self.fc(decoder_out.squeeze(1))
            output = output.view(batch_size,-1)
            
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
           
            
            #end if <EOS detected>
            if idx2word[predicted_word_idx.item()] == "<EOS>":
                break
            if predicted_word_idx.item() not in [word2idx['<SOS>'], word2idx['<PAD>']]:
                 captions.append(predicted_word_idx.item())
            #send generated word as the next caption
            src_token = torch.tensor(predicted_word_idx.item()).view(1,-1).cuda()
            tgt_token = torch.tensor(word2idx['<PAD>']).view(1,-1).cuda()
        #covert the vocab idx to words and return sentence
        return [idx2word[idx] for idx in captions]
    
    
    
