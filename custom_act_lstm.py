from mimetypes import init
import torch
import torch.nn as nn
import  torch.nn.utils.rnn as rnn_u
from torch.nn import functional as F


class CustomActivatedLSTMCell(nn.Module):
    def __init__(self,input_size,hidden_size, act_func:object) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act_func = act_func # custom activation function
        self.w_ih = nn.Parameter(torch.rand(4*hidden_size,input_size))  #weight per input - mapped to hidden size
        self.w_hh = nn.Parameter(torch.rand(4*hidden_size, hidden_size)) 
        self.b_ih = nn.Parameter(torch.rand(4*hidden_size))
        self.b_hh = nn.Parameter(torch.rand(4*hidden_size))

    def forward(self,x:torch.Tensor,state:tuple): 
        ht_1, ct_1 = state
        device = x.device
        self.w_ih = self.w_ih.to(device=device)
        self.w_hh = self.w_hh.to(device=device)
        self.b_ih = self.b_ih.to(device=device)
        self.b_hh = self.b_hh.to(device=device)

        w_xf, w_xi, w_xic, w_xo = torch.chunk(self.w_ih,4,0,)
        w_hf, w_hi, w_hic, w_ho = torch.chunk(self.w_hh,4,0)
        b_xf, b_xi, b_xic, b_xo = torch.chunk(self.b_ih,4,0)
        b_hf, b_hi, b_hic, b_ho = torch.chunk(self.b_hh,4,0)
        
        #forget gate - reassigns ct_1
        ft_inputs = F.linear(x,w_xf,b_xf) + F.linear(ht_1,w_hf,b_hf)
        ft = torch.sigmoid(ft_inputs)
        ct_forget = ct_1*ft

        #input gate
        it_inputs = F.linear(x,w_xi,b_xi) + F.linear(ht_1,w_hi,b_hi)
        it = torch.sigmoid(it_inputs)
        
        #input candidates - new ct
        ic_inputs = F.linear(x,w_xic,b_xic) + F.linear(ht_1,w_hic,b_hic)
        i_cands = self.act_func(ic_inputs)
        ct:torch.Tensor = ct_forget + (i_cands*it)

        #output gate
        ot_inputs = F.linear(x,w_xo,b_xo) + F.linear(ht_1,w_ho,b_ho)
        ot = torch.sigmoid(ot_inputs)

        #output candidates
        o_cands = self.act_func(ct)
        ht:torch.Tensor = ot*o_cands

        return ht, ct
    
class CustomLSTMLayer(nn.Module):
        def __init__(self, input_size, hidden_size, activation, batch_first = True):

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.activation = activation
            self.LSTMcell =  CustomActivatedLSTMCell(input_size=self.input_size, hidden_size=self.hidden_size, act_func=self.activation)
            self.batch_first = batch_first

        def forward(self, x:torch.Tensor, init_state = None):
            if self.batch_first:
                batch_size = x.shape(0)

            if init_state is not None: 
                ht, ct = init_state
                #verify state shapes
                if ht.shape(0) != batch_size or ht.shape(1)!= self.hidden_size:
                    raise ValueError(f"Init state 'ht' has shape {ht.shape}, while requiring shape {[batch_size,self.hidden_size]}")
                if ct.shape(0) != batch_size or ct.shape(1)!= self.hidden_size:
                    raise ValueError(f"Init state 'ct' has shape {ct.shape}, while requiring shape {[batch_size,self.hidden_size]}")
            else:
                ht = torch.zeros(batch_size, self.hidden_size,device=x.device)
                ct = torch.zeros(batch_size, self.hidden_size,device=x.device)

            outputs = []
            for t in range(x.shape(1)): #recurrent through seq_len
                xt = x[:, t, :]
                ht:torch.Tensor; ct:torch.Tensor = self.LSTMcell(xt,(ht,ct))
                outputs.append(ht.unsqueeze(1)) #[batch_size, 1, hidden_size]

            outputs = torch.cat(outputs)
            return outputs, (ht,ct)
    