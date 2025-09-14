import torch
import torch.nn as nn

class soilTransformer(nn.Module):
    '''
    Purpose:
    An autoencoder with transformer layers
    positional encoding is learned, not prescribed
    '''
        
    def __init__(self, input_dim, embed_dim, seq_len, num_heads = 4, num_layers = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # learned positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        print(f"self.positional_encoding shape: {self.positional_encoding.shape}")
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, input_dim)
        )
   
    def getEmbeddings(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1)]
        encoded = self.encoder(x)
        return encoded
    
    def forward(self, x):
        embeddings = self.getEmbeddings( x)
        return self.decoder(embeddings)
    
    def reconstructByLatent(self, embeddings):
        return self.decoder(embeddings)
    

class soilTransformer2(nn.Module):
    '''
    Purpose:
    An autoencoder with transformer layers
    Positional encoding is prescribed, not learned
    Enable using mask to excluding padded positions
    
    '''
    def __init__(self, input_dim, embed_dim, num_heads = 4, num_layers = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)

        # depth-based posiitonal encoding
        self.positional_encoding = DepthPositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        self.decoder = nn.Sequential(nn.Linear(embed_dim, embed_dim), 
            nn.ReLU(), 
            nn.Linear(embed_dim, input_dim)
            )
        
    def getEmbeddings(self, x):
        depth = x[:,:, [0]]

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # if x does not have depth (first column), then just use index as depth
        if len(x.shape) == 2:
            depth = torch.arange(0, seq_len).unsequeeze(0).repeat(batch_size, 1)


        features = self.embedding(x)

        # integrate positional encoding
        x = features + self.positional_encoding(depth)
        return self.encoder(x)

    def forward(self, x):
        embeddings = self.getEmbeddings(x)
        return self.decoder(embeddings)


class DepthPositionalEncoding(nn.Module):
    """
    Depth-based positional encoding module
    Use sinusoidal encoding but takes depth values as input
    """        

    def __init__(self, embed_dim:int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, depth: torch.Tensor):
        '''
        depth: shape [batch, seq_len, 1] (first dimension is depth)
        returns: positional encoding of shape [batch, seq_len, embed_dim]
        '''

        device = depth.device
        batch, seq_len,_ = depth.shape

        # [seq_len, embed_dim/2]
        div_term = torch.exp(torch.arange(0, self.embed_dim, step = 2, device = device) *
                             (-math.log(10000.0) / self.embed_dim))

        # expand for broadcasting [batch, seq_len, 1] -> [batch, seq_len, embed_dim /2]
        depth_expanded = depth.repeat(1, 1, self.embed_dim // 2) 

        pe_sin = torch.sin(depth_expanded * div_term)
        pe_cos = torch.cos(depth_expanded * div_term)

        # interleave sin and cos to [batch, seq_len, embed_dim]
        pe = torch.zeros(batch, seq_len, self.embed_dim, device = device)
        pe[:, :, 0::2] = pe_sin
        pe[:, :, 1::2] = pe_cos

        return pe


class soilTransformer3(nn.Module):
    '''
    Purpose:
    An autoencoder with transformer layers
    positional encoding is learned, not prescribed
    Enable using mask to excluding padded positions
    
    '''
    def __init__(self, input_dim, embed_dim, seq_len, num_heads = 4, num_layers = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # learned positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        print(f"self.positional_encoding shape: {self.positional_encoding.shape}")
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, input_dim)
        )
   
    def getEmbeddings(self, x, src_key_padding_mask = None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        # Note: TransformerEncoder's forward arguments are {src, mask = None, src_key_padding_mask = None}
        encoded = self.encoder(x, mask = None, src_key_padding_mask = src_key_padding_mask)
        return encoded
    
    def forward(self, x, src_key_padding_mask = None):
        embeddings = self.getEmbeddings(x, src_key_padding_mask = src_key_padding_mask)
        return self.decoder(embeddings)
    
    def reconstructByLatent(self, embeddings):
        return self.decoder(embeddings)
    
    def getOriginalEmbeddings(self, x):
        return  self.embedding(x)