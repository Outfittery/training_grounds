from enum import Enum

class Dim3NetworkType(Enum):
    LSTM = 0
    AlonAttention = 1
    AlonAttentionSigmoid = 2
    AlonAttentionWithoutFullyConnected = 3
    AlonAttentionWithoutFullyConnectedSigmoid = 4
    SelfAttentionAndLSTM = 5