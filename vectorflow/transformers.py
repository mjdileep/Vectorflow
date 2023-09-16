from .layers import Layer,  Dropout, Linear, Stack, ReLU, Layernorm, CustomModule
import numpy as np
from .loss_functions import softmax


class CausalSelfAttentionHead(Layer):
    """
       Causal Self Attention Head
    """
    
    def __init__(self, n_embed: int, head_size: int, block_size:int , keep:float=0.8):
        """
            Constructor of Liniear Layer 
            Inputs:
            - n_embed: (int) Dimentions of token embeddings 
            - head_size: Size of a single head (equal to n_embed if you are using single head attention)
            - block_size: Attention block size (temporal span of past tokens)
        """
        super().__init__()
        # You can actually combine these 3 yet it becomes hard to calculate dradient manually after
        self.key = Linear(n_embed, head_size, bias=False)
        self.value = Linear(n_embed, head_size, bias=False)
        self.query = Linear(n_embed, head_size, bias=False)
        self.tril = np.tril(np.ones((block_size,block_size)))
        self.dropout = Dropout(keep=keep)
        self.cache = None
        
    def forward(self, x, **kwargs):
        """_summary_

        Args:
            x (np.array): data vector

        Returns:
            _type_: _description_
        """
        B, T, C = x.shape 
        key = self.key(x) # B, T, head_size
        query = self.query(x)
        scale = C ** -0.5 
        # Compute attention scores ("affinities") 
        wei = key @ np.transpose(query, (0, -1, -2))   * scale#B, T, T (scaled to unity due to inceased variance will push the softmax scores toward one-hot)
        wei = np.ma.array(wei, mask=np.broadcast_to(self.tril[:T, :T]==0, (B, T, T))).filled(fill_value=-np.inf) # Masking future tokens
        # Take the softmax distribution of affinities
        wei_exp = np.exp(wei)
        wei_exp_sum = wei_exp.sum(axis=-1, keepdims=True)
        wei_softmax = wei_exp / (wei_exp_sum+0.00000001)

        wei_softmax = self.dropout(wei_softmax, **kwargs) # Drop some connections during the traning 
        
        value = self.value(x)
        out = wei_softmax @ value # Retrive values 

        # Cache the values for backprop calculations
        self.cache = {"wei":wei, "value":value, "query":query,
                      "key":key, 'wei_softmax':wei_softmax, "scale":scale}
        
        return out
    
    def backward(self, dx):
        dwei_softmax = dx @ np.transpose(self.cache['value'], (0, -1, -2)) 
        wei_softmax = self.cache["wei_softmax"] # B, T, T
        
        dvalue = wei_softmax @ dx
        dvx = self.value.backward(dvalue)

        dwei_softmax = self.dropout.backward(dwei_softmax)
        
        dwei = dwei_softmax*wei_softmax*(1-wei_softmax)*self.cache["scale"]
        
        dkey = dwei @ self.cache["query"]
        dkx = self.key.backward(dkey)

        dquery = dwei @ self.cache["key"]
        dqx = dkx = self.key.backward(dquery)
        dout = np.transpose(dvx+dkx+dqx, (0, -2, -1))
        return dout
    
    def step(self, fn, lr):
        self.key.step(fn, lr)
        self.value.step(fn, lr)
        self.query.step(fn, lr)
    
    def grad_zero(self):
        """
            Set accumilated gradients to zero in the layer 
        """
        self.key.grad_zero()
        self.value.grad_zero()
        self.query.grad_zero()
    




class MultiHeadAttention(Layer):

    def __init__(self, n_embed, n_heads, head_size, block_size, keep=0.8):
        """_summary_

        Args:
            n_embed (_type_): _description_
            n_heads (_type_): _description_
            head_size (_type_): _description_
            block_size (_type_): _description_
            keep (float, optional): _description_. Defaults to 0.8.
        """
        super().__init__()
        self.n_heads = n_heads
        self.heads = [CausalSelfAttentionHead(n_embed, head_size, block_size, keep=0.8) for _ in range(n_heads)]
        self.proj = Linear(head_size*n_heads, n_embed) # 256 32*8
        self.dropout = Dropout(keep)
        self.cache = None

    def forward(self, x, **kwargs):
        attention_outputs = np.concatenate([h(x, **kwargs) for h in self.heads], axis=-1)
        projected = self.proj(attention_outputs)
        out = self.dropout(projected, **kwargs)
        return out

    def backward(self, dx):
        dprojection_outputs = self.dropout.backward(dx)
        dattention_outputs = self.proj.backward(dprojection_outputs)
        dout = np.sum([h.backward(dout) for h, dout in zip(self.heads, np.split(dattention_outputs, self.n_heads, axis=-1))])
        return dout
    
    def step(self, fn, lr):
        for h in self.heads:
            h.step(fn, lr)
        self.proj.step(fn, lr)

    def grad_zero(self):
        """
            Set accumilated gradients to zero in the layer 
        """
        for h in self.heads:
            h.grad_zero()
        self.proj.grad_zero()
    

class FeedForward(CustomModule):
    def __init__(self, n_embed, keep=0.8):
        super().__init__()
        self.net = Stack(
            Linear(n_embed, 4*n_embed),
            ReLU(),
            Linear(4*n_embed, n_embed),
            Dropout(keep),
        )
    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)
    
    def backward(self, dx):
        return self.net.backward(dout=dx)

    def step(self, fn, lr):
        self.net.step(fn, lr)
    
    def grad_zero(self):
        """
            Set accumilated gradients to zero in the layer 
        """
        self.net.grad_zero()


class Block(CustomModule):

    def __init__(self, n_embed, n_heads, block_size, keep=0.8):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_embed=n_embed, n_heads=n_heads, head_size=head_size,
                                     block_size=block_size, keep=keep)
        self.ffwd = FeedForward(n_embed, keep=0.8)
        self.ln1 = Layernorm(n_embed)
        self.ln2 = Layernorm(n_embed)

    def forward(self, x, **kwargs):
        x = self.ln1(x)
        x = x + self.sa(x, **kwargs) # Residual connections 
        x = self.ln2(x)
        x = x + self.ffwd(x, **kwargs)  # Residual connections 
        return x 
    
    def backward(self, dx):
        dffwd = self.ffwd.backward(dx)
        dln2 = self.ln2.backward(dx+dffwd)
        dsa = self.sa.backward(dln2)
        dln1 = self.ln1.backward(dln2+dsa)
        return dln1


    def step(self, fn, lr):
        self.sa.step(fn, lr)
        self.ffwd.step(fn, lr)
        self.ln1.step(fn, lr)
        self.ln2.step(fn, lr)
    
    def grad_zero(self):
        """
            Set accumilated gradients to zero in the layer 
        """
        self.sa.grad_zero()
        self.ffwd.grad_zero()
        self.ln1.grad_zero()
        self.ln2.grad_zero()


class Embedding(Layer):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.params["w"] = np.random.randn(vocab_size, n_embed)* np.sqrt(2/(n_embed**2)) # He initialization
        self.grads["w"] = np.zeros((vocab_size, n_embed))
        self.param_configs["w"] = {}
        self.cache = None

    def forward(self, x, **kwargs):
        B, T = x.shape
        x = x.flatten()
        out = self.params["w"][x]
        self.cache = x
        return out.reshape(B, T, -1)
    
    def backward(self, dx):
        B, T, _ = dx.shape
        dx = dx.reshape(B*T, -1)
        self.grads["w"][self.cache] += dx
        return dx.reshape(B, T, -1)


class Transformer(CustomModule):
    def __init__(self, vocab_size: int, n_embed: int, block_size: int, n_heads: int, n_layers: int, keep=0.8):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = Embedding(vocab_size, n_embed)
        self.position_embedding_table = Embedding(block_size, n_embed)
        self.blocks = Stack(*[Block(n_embed, n_heads, block_size, keep) for _ in range(n_layers)])
        self.ln_f = Layernorm(n_embed)
        self.lm_head = Linear(n_embed, vocab_size)

    def forward(self, idx, **kwargs):
        B, T = idx.shape
        targets = None
        if "targets" in kwargs:
            targets = kwargs.pop("targets")
        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(np.arange(T).reshape(1, -1)) # T,C
        x = tok_emb + pos_emb
        x = self.blocks(x, **kwargs)
        x = self.ln_f(x, **kwargs)
        logits = self.lm_head(x, **kwargs)
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.reshape(B*T,C)
            targets = targets.reshape(B*T)
            loss, dout = softmax(logits, targets)
            return logits.reshape(B, T,-1), loss, dout.reshape(B, T,-1)
        else:
            return logits, None, None
    
    def backward(self, dx, print_grad = True):
        if print_grad:
            print("Gradient Recieved:", np.abs(dx).sum())
        dx = self.lm_head.backward(dx)
        if print_grad:
            print("Gradient after LM Head:", np.abs(dx).sum())
        dx = self.ln_f.backward(dx)
        if print_grad:
            print("Gradient after LN:", np.abs(dx).sum())
        dx = self.blocks.backward(dx)
        if print_grad:
            print("Gradient after blocks:", np.abs(dx).sum())
            print("---------------------------------------\n")
        self.token_embedding_table.backward(dx)
        self.position_embedding_table.backward(np.sum(dx, axis=0, keepdims=1))
    
    def step(self, fn, lr):
        self.lm_head.step(fn, lr)
        self.ln_f.step(fn, lr)
        self.blocks.step(fn, lr)
        self.token_embedding_table.step(fn, lr)
        self.position_embedding_table.step(fn, lr)


    def generate(self, idx, max_new_tokens):
        B, T = idx.shape
        assert B ==1, "Only supports 1 batch at a time"
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _, _ = self(idx_cond, train=False)
            logits = logits[:,-1,:]
            exp_logits = np.exp(logits)
            prob = exp_logits/exp_logits.sum(axis=-1)
            idx_next =  np.array(np.argmax(np.random.multinomial(10, prob.reshape(-1), size=1))).reshape(1,1)
            idx = np.concatenate((idx, idx_next), axis=1)
        return idx
    
    def grad_zero(self):
        """
            Set accumilated gradients to zero in the layer 
        """
        self.lm_head.grad_zero()
        self.ln_f.grad_zero()
        self.blocks.grad_zero()
        self.token_embedding_table.grad_zero()
        self.position_embedding_table.grad_zero()
        
