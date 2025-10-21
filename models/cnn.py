import torch
from torch import nn

class CNN(nn.module):
    def __init__(self,dropout: float = 0.0,use_batchnorm: bool = False,
                 in_channels: int = 1, num_classes:int = 10):
        super().__init__()

        c1,c2 = 5,10
        layers= []
        layers += [nn.Conv2d(in_channels,c1,kernel_size=3,padding=1),# 28x28
                   nn.ReLU(inplace=True)]
        if use_batchnorm:
            layers += [nn.BatchNorm2d(c1)]
        layers += [nn.MaxPool2d(2,2)] #28x28 -> 14x14
        layers +=[nn.Conv2d(c1,c2,kernel_suze=3,padding=1),
                  nn.ReLU(inplace=True)]#14x14
        if use_batchnorm:
            layers += [nn.BatchNorm2d(c2)]
        layers += [nn.MaxPool2d(2,2)]#14x14 -> 7x7
        #salviamo tutto ciò che abbiamo fatto sopra in self.feature e con Sequential passiamo tutti i layer
        self.feature = nn.Sequential(*layers)
         #il dropout aggiunge 0 randomici che il modello tende ad imparare per diminuire l'overfitting
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        #per trasformazione lineare intendiamo il dato per il peso e fa la somma con gli altri dati (x1*w1 + x2*w2...)
        #7*7*c2 perchè dopo il maxpooling l'immagine sarà ridotta a 7x7 e c2 sono i canali
        self.classifier = nn.Linear(7*7*c2,num_classes)#serie di layer connessi lineari

    #classe di pytorch per il forward pass
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)#richiama le features sopra
        x = torch.flatten(x,1)#converte la matrice in vettore sulle righe
        x = self.dropout(x)#richiama il dropout sopra
        x = self.classifier(x)#richiama il classificatore nella classe
        return x

