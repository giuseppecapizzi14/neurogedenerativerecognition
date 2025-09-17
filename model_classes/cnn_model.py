import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Dropout, Linear, MaxPool1d, Module, ReLU, Sequential, Softmax


class CNNModel(Module):
    # Strati convoluzionali per estrarre le features rilevanti
    feature_extraction: Sequential

    # Strati fully-connected lineari di classificazione
    classification: Sequential

    def __init__(self, waveform_size: int, dropout: float, device: torch.device):
        def output_size(input_size: int, padding: int, kernel_size: int, stride: int) -> int:
            return (input_size + 2 * padding - kernel_size) // stride + 1

        super(CNNModel, self).__init__() # type: ignore

        self.feature_extraction = Sequential(
            # Primo strato convoluzionale - pattern temporali grossolani
            Conv1d(in_channels = 1, out_channels = 16, kernel_size = 15, stride = 6, padding = 0, device = device),
            BatchNorm1d(num_features = 16, device = device),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 4, stride = 3, padding = 0),
            Dropout(dropout),

            # Secondo strato convoluzionale - features intermedie
            Conv1d(in_channels = 16, out_channels = 32, kernel_size = 9, stride = 4, padding = 0, device = device),
            BatchNorm1d(num_features = 32, device = device),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 2, padding = 0),
            Dropout(dropout),

            # Terzo strato convoluzionale - features finali dettagliate
            Conv1d(in_channels = 32, out_channels = 48, kernel_size = 5, stride = 2, padding = 0, device = device),
            BatchNorm1d(num_features = 48, device = device),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 2, stride = 2, padding = 0),
            Dropout(dropout)
        )

        # Calcoliamo la dimensione dell'output di tutti gli strati convoluzionali
        sample_len = waveform_size

        # Teniamo traccia dell'ultimo strato convoluzionale per calcolare la dimensione dell'input
        # degli strati fully-connected
        last_conv_layer: Conv1d | None = None

        for layer in self.feature_extraction.modules():
            match layer:
                case Conv1d():
                    last_conv_layer = layer

                    padding: int = layer.padding[0] # type: ignore
                    kernel_size = layer.kernel_size[0]
                    stride = layer.stride[0]

                    sample_len = output_size(sample_len, padding, kernel_size, stride)
                case MaxPool1d():
                    padding: int = layer.padding # type: ignore
                    kernel_size: int = layer.kernel_size # type: ignore
                    stride: int = layer.stride # type: ignore

                    sample_len = output_size(sample_len, padding, kernel_size, stride)
                case _:
                    pass

        assert not last_conv_layer is None, "At least one convolutional layer must be present"

        self.classification = Sequential(
            # Primo strato completamente connesso - estrazione features complesse
            Linear(in_features = last_conv_layer.out_channels * sample_len, out_features = 128, device = device),
            BatchNorm1d(num_features = 128, device = device),
            ReLU(inplace = True),
            Dropout(dropout),

            # Secondo strato completamente connesso - raffinamento
            Linear(in_features = 128, out_features = 64, device = device),
            ReLU(inplace = True),
            Dropout(dropout),

            # Strato di output
            Linear(in_features = 64, out_features = 2, device = device),
            Softmax(1)
        )

    def forward(self, x: Tensor):
        # Passaggio attraverso gli strati convoluzionali
        x = self.feature_extraction(x)

        # Riformatta l'output per il passaggio attraverso i layer completamente connessi
        x = x.flatten(1)

        # Passaggio attraverso gli strati completamente connessi
        x = self.classification(x)

        return x
