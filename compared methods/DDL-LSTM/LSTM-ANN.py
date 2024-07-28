# torch
import torch
from torch import  nn
from torch.autograd import Variable


class LSTM_ANN(nn.Module):

    def __init__(self, num_targets, input_size, hidden_size, num_layers):
        super(LSTM_ANN, self).__init__()
        
        #params
        self.num_targets = num_targets
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        #LSTM module
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        
        #Predictor ANN
        self.pann_fc_1 = nn.Linear(self.hidden_size, self.hidden_size)      #FC 1  
        self.pann_fc_2 = nn.Linear(self.hidden_size, self.hidden_size)      #FC 2
        self.pann_fc_3 = nn.Linear(self.hidden_size, self.num_targets)      #FC target map
        self.pann_nl = nn.ReLU()
        
        #Attention ANN
        self.aann_fc_1 = nn.Linear(self.hidden_size, self.hidden_size)      #FC 1  
        self.aann_fc_2 = nn.Linear(self.hidden_size, self.hidden_size)      #FC 2  
        self.aann_fc_3 = nn.Linear(self.hidden_size,1)                      #FC attn map
        self.aann_nl_1 = nn.ReLU()        
        self.aann_nl_2 = nn.Softmax(dim=0)
 
    def forward(self, x):
       
       
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))                     
        h_out = ula.view(-1, self.hidden_size)
        
        
        #prediction model
        pred_in = self.pann_nl(h_out)
        final_out_pred = self.pann_nl(self.pann_fc_1(pred_in))
        final_out_pred = self.pann_fc_2(final_out_pred)       
        out_pred = self.pann_fc_3((final_out_pred))       
        
        
        #attention model       
        attn_in = self.aann_fc_1(h_out)   
        final_out_attn = self.aann_fc_2(self.aann_nl_1(attn_in))
        final_out_attn = self.aann_nl_1(attn_in)
        attn = self.aann_nl_2(self.aann_fc_3(final_out_attn))  #softmax   
        
        
        #combine predictions
        out_pred = out_pred.transpose(0,1)
        attn_final = attn.transpose(0,1).repeat(self.num_targets,1)           
        final_out  = torch.mul(out_pred,attn_final).sum(dim=1)  

        
        return [final_out,attn_final]
