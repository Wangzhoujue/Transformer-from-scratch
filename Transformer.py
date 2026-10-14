##导入依赖包
import torch
from torch import nn
import torch.nn.functional as F
import math


def prepare_mask(mask,B):
    if mask.dim()==1:
        return mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    elif mask.dim()==2:
        if mask.shape[0] == B:
            # Padding Mask
            # [B,T] → [B,1,1,T]
            return mask.unsqueeze(1).unsqueeze(2)

        else:
            # Causal Mask
            # [T,T] → [1,1,T,T]
            return mask.unsqueeze(0).unsqueeze(1)
        return mask.unsqueeze(0).unsqueeze(1)

    return mask



class Embedding(nn.Module):
    def __init__(self, vocab_size,embedding_dim):
        super(Embedding,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.weight=nn.Parameter(torch.randn(self.vocab_size,self.embedding_dim))

    def forward(self, ids):
        return self.weight[ids]

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding,self).__init__()
        self.max_len=max_len
        self.d_model=d_model
        pe=torch.zeros(self.max_len,self.d_model)
        Position=torch.arange(0,self.max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,self.d_model,2) * -math.log(10000.0)/self.d_model)
        pe[:,0::2]=torch.sin(Position*div_term)
        pe[:,1::2]=torch.cos(Position*div_term)
        self.register_buffer("pe",pe)

    def forward(self,embedding):
        seq_len=embedding.size(1)
        return embedding+self.pe[:seq_len]


##构建多头注意力类
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_head,dropout=False):
        super(MultiHeadAttention,self).__init__()
        self.num_head=num_head
        self.d_model=d_model
        assert d_model%num_head==0
        self.head_dim=d_model//num_head
        self.q_proj=nn.Linear(d_model,d_model)
        self.k_proj=nn.Linear(d_model,d_model)
        self.v_proj=nn.Linear(d_model,d_model)
        self.out_proj=nn.Linear(d_model,d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self,Q,K,V,mask):
        B=Q.shape[0]
        Q_S=Q.shape[1]
        K_S=K.shape[1]
        Q=self.q_proj(Q)
        K=self.k_proj(K)
        V=self.v_proj(V)
        Q=(Q.reshape(B,Q_S,self.num_head,self.head_dim)).transpose(1,2)
        K=(K.reshape(B,K_S,self.num_head,self.head_dim)).transpose(1,2)
        V=(V.reshape(B,K_S,self.num_head,self.head_dim)).transpose(1,2)
        head_attn_scores=Q @ K.transpose(-2,-1)/math.sqrt(self.head_dim)
        if mask is not None:
            mask = prepare_mask(mask,B)
            head_attn_scores=head_attn_scores.masked_fill(
                mask==0,
                -1e9
            )
        head_attn_weight=self.attn_dropout(torch.softmax(head_attn_scores,dim=-1))
        head_output=head_attn_weight @ V
        output=(head_output.transpose(1,2)).contiguous().view(B,Q_S,self.d_model)

        return self.out_proj(output)

##前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model,d_ff,dropout=False):
        super(FeedForward,self).__init__()
        self.up_proj=nn.Linear(d_model,d_ff)
        self.down_proj=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,X):
        return self.dropout(self.down_proj(F.relu(self.up_proj(X))))

    ##归一化层
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-6):
        super(LayerNorm,self).__init__()

        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
        self.eps=eps

    def forward(self,X):
        mean=X.mean(dim=-1, keepdim=True)
        variance = X.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (X - mean) / torch.sqrt(variance + self.eps)
        return normalized *self.gamma +self.beta

    ##定义编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff):
        super(EncoderLayer,self).__init__()
        self.Attention=MultiHeadAttention(d_model,num_head)
        self.FeadForward=FeedForward(d_model,d_ff)
        self.Norm_1=LayerNorm(d_model)
        self.Norm_2=LayerNorm(d_model)
    
    def forward(self, X,mask):
        Attention_output=self.Attention(X,X,X,mask)
        X=self.Norm_1(X+Attention_output)
        FFN_output=self.FeadForward(X)
        X=self.Norm_2(X+FFN_output)
        return X

##定义编码器
class Encoder(nn.Module):
    def __init__(self,num_Layers,d_model, num_head, d_ff):
        super(Encoder,self).__init__()
        self.Layers=nn.ModuleList()
        for i in range(num_Layers):
            encoderLayer=EncoderLayer(d_model,num_head,d_ff)
            self.Layers.append(encoderLayer)
        
    def forward(self,X,mask):
        for Layer in self.Layers:
            X=Layer(X,mask)
        return X


## 构建解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff):
        super(DecoderLayer,self).__init__()
        self.Self_Attention=MultiHeadAttention(d_model,num_head)
        self.Cross_Attention=MultiHeadAttention(d_model,num_head)
        self.FFN=FeedForward(d_model,d_ff)
        self.Norm_1=LayerNorm(d_model)
        self.Norm_2=LayerNorm(d_model)
        self.Norm_3=LayerNorm(d_model)

    def forward(self,Decoder_X,Encoder_Output,Self_mask,Cross_mask):
        Self_Attention_Output=self.Self_Attention(Decoder_X,Decoder_X,Decoder_X,Self_mask)
        X=self.Norm_1(Decoder_X+Self_Attention_Output)
        Cross_Attention_Output=self.Cross_Attention(X,Encoder_Output,Encoder_Output,Cross_mask)
        X=self.Norm_2(X+Cross_Attention_Output)
        FFN_Output=self.FFN(X)
        Output=self.Norm_3(X+FFN_Output)
        return Output


#构建解码器
class Decoder(nn.Module):
    def __init__(self,num_Layers,d_model, num_head, d_ff):
        super(Decoder,self).__init__()
        self.Layers=nn.ModuleList()
        for i in range(num_Layers):
                    decoderLayer=DecoderLayer(d_model,num_head,d_ff)
                    self.Layers.append(decoderLayer)

    def forward(self,Decoder_X,Encoder_Output,Self_mask,Cross_mask):
        for Layer in self.Layers:
              Decoder_X=Layer(Decoder_X,Encoder_Output,Self_mask,Cross_mask)
        return Decoder_X


           
        

##定义生成器
class Generator(nn.Module):
    def __init__(self, d_model,vocab_size):
        super(Generator,self).__init__()
        self.Out_proj=nn.Linear(d_model,vocab_size)


    def forward(self,Decoder_Output):
        return self.Out_proj(Decoder_Output)



##定义Transformer类
class Transformer(nn.Module):
    def __init__(self,d_model,vocab_size,num_head,d_ff,num_Layers,max_len):
        super(Transformer,self).__init__()
        self.Input_Embedding=Embedding(vocab_size,d_model)
        self.Output_Embedding=Embedding(vocab_size,d_model)
        self.PositionEncoding=PositionalEncoding(max_len,d_model)
        self.Encoder=Encoder(num_Layers,d_model,num_head,d_ff)
        self.Decoder=Decoder(num_Layers,d_model,num_head,d_ff)
        self.Generator=Generator(d_model,vocab_size)

    def forward(self,src,tgt,src_mask,tgt_mask):
        Input_embedding=self.PositionEncoding(self.Input_Embedding(src))
        Output_embedding=self.PositionEncoding(self.Output_Embedding(tgt[:,:-1]))
        Encoder_output=self.Encoder(Input_embedding,src_mask)
        Decoder_output=self.Decoder(Output_embedding,Encoder_output,tgt_mask,src_mask)
        Logit=self.Generator(Decoder_output)

        return Logit