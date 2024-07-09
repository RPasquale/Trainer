import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import inspect

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the VAE (Vision Model)
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

# Define the MDN-RNN (Memory Model)
class MDNRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gaussians):
        super(MDNRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_gaussians * output_dim * 3)
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim

    def forward(self, x, h):
        out, h = self.rnn(x.unsqueeze(0), h)  # Ensure x has a batch dimension
        out = self.fc(out)
        return out, h

    def get_mixture_coef(self, y):
        y = y.view(-1, self.num_gaussians * self.output_dim * 3)
        mean = y[:, :self.num_gaussians * self.output_dim]
        mean = mean.view(-1, self.num_gaussians, self.output_dim)
        log_sigma = y[:, self.num_gaussians * self.output_dim:2 * self.num_gaussians * self.output_dim]
        log_sigma = log_sigma.view(-1, self.num_gaussians, self.output_dim)
        log_pi = y[:, 2 * self.num_gaussians * self.output_dim:]
        log_pi = log_pi.view(-1, self.num_gaussians)
        return mean, log_sigma, log_pi

    def loss_function(self, y, mu, log_sigma, log_pi):
        y = y.unsqueeze(1).repeat(1, self.num_gaussians, 1)
        log_gauss = -0.5 * ((y - mu) ** 2 / torch.exp(log_sigma) + log_sigma + math.log(2 * math.pi))
        log_gauss = torch.sum(log_gauss, dim=2)
        log_gauss = log_gauss + log_pi
        loss = -torch.logsumexp(log_gauss, dim=1)
        return torch.mean(loss)

# Define the Node class for MCTS
class Node:
    def __init__(self, prompt, text, parent=None):
        self.prompt = prompt
        self.text = text
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.value = 0
        self.policy = None
        self.hidden_state = None

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child_node):
        self.children.append(child_node)

# MCTS functions
def ucb1(node, exploration_param=1.41):
    if node.visits == 0:
        return float('inf')
    return node.reward / node.visits + exploration_param * np.sqrt(np.log(node.parent.visits) / node.visits)

def select_node(node):
    best_value = -float('inf')
    best_node = None
    for child in node.children:
        ucb_value = ucb1(child)
        if ucb_value > best_value:
            best_value = ucb_value
            best_node = child
    return best_node

def expand_node(node, tokenizer, gpt_model, vae, mdn_rnn, num_expansions=5):
    inputs = tokenizer(node.text, return_tensors='pt', padding=True, truncation=True).to(device)
    attention_mask = inputs['attention_mask']
    input_ids = inputs['input_ids']
    outputs = gpt_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20, num_return_sequences=num_expansions, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    for output in outputs:
        new_text = tokenizer.decode(output, skip_special_tokens=True)
        child_node = Node(node.prompt, new_text, parent=node)
        node.add_child(child_node)

        # Encode the text with the VAE
        embedding = get_embedding(new_text, st_model).detach().cpu().numpy()
        z, logvar = vae.encode(torch.tensor(embedding, dtype=torch.float32).to(device))
        z = vae.reparameterize(z, logvar).unsqueeze(0)

        # Use MDN-RNN to get the next hidden state and reward
        if node.hidden_state is None:
            h = (torch.zeros(1, mdn_rnn.hidden_dim).to(device), torch.zeros(1, mdn_rnn.hidden_dim).to(device))
        else:
            h = node.hidden_state

        _, h = mdn_rnn(z, (h[0].unsqueeze(0), h[1].unsqueeze(0)))
        h = (h[0].squeeze(0), h[1].squeeze(0))
        child_node.hidden_state = h

def get_embedding(text, model):
    return model.encode(text, convert_to_tensor=True)

def get_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    max_length = model.config.n_positions
    stride = 512
    lls = []

    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()

def simulate(node, gpt_model, st_model, tokenizer, rewards, vae, mdn_rnn):
    reward = 0
    for reward_func in rewards:
        reward += reward_func(node, gpt_model, st_model, tokenizer)
    node.reward += reward
    node.visits += 1

    # Encode the text with the VAE
    embedding = get_embedding(node.text, st_model).detach().cpu().numpy()
    z, logvar = vae.encode(torch.tensor(embedding, dtype=torch.float32).to(device))
    z = vae.reparameterize(z, logvar).unsqueeze(0)

    # Use MDN-RNN to get the next hidden state and reward
    if node.hidden_state is None:
        h = (torch.zeros(1, mdn_rnn.hidden_dim).to(device), torch.zeros(1, mdn_rnn.hidden_dim).to(device))
    else:
        h = node.hidden_state

    next_state, h = mdn_rnn(z, (h[0].unsqueeze(0), h[1].unsqueeze(0)))
    h = (h[0].squeeze(0), h[1].squeeze(0))
    next_state = next_state.squeeze(0)  # Removing the batch dimension
    node.hidden_state = h  # Update the node's hidden state

    mean, log_sigma, log_pi = mdn_rnn.get_mixture_coef(next_state)
    log_pi = log_pi.unsqueeze(-1)  # Align dimensions for broadcasting
    immediate_reward = torch.sum(mean * torch.exp(log_pi), dim=1)
    node.reward += immediate_reward.mean().item()  # Take the mean reward

    return reward

def backpropagate(node, reward):
    while node is not None:
        node.reward += reward
        node.visits += 1
        node = node.parent

def relevance_reward(node, gpt_model, st_model, tokenizer):
    original_embedding = get_embedding(node.prompt, st_model)
    generated_embedding = get_embedding(node.text, st_model)
    return util.pytorch_cos_sim(original_embedding, generated_embedding).item()

def coherence_reward(node, gpt_model, st_model, tokenizer):
    return -get_perplexity(node.text, gpt_model, tokenizer)

# Define the MoE Layer
class MoELayer(nn.Module):
    def __init__(self, num_experts, expert_dim, input_dim):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        routing_weights = F.softmax(self.router(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.sum(routing_weights.unsqueeze(-1) * expert_outputs, dim=-1)
        return output

# Define the Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, model_dim, num_timesteps):
        super(DiffusionModel, self).__init__()
        self.model = nn.Transformer(d_model=model_dim)
        self.timesteps = num_timesteps

    def forward(self, x, t):
        noise = torch.randn_like(x) * (t / self.timesteps)
        noisy_input = x + noise
        output = self.model(noisy_input)
        return output

    def compute_loss(self, x, t):
        noisy_input = x + torch.randn_like(x) * (t / self.timesteps)
        output = self.model(noisy_input)
        loss = F.mse_loss(output, x)
        return loss

# Define the GPT components
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.scale_init = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.scale_init = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'scale_init'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        print("Keys in the HuggingFace model but not in the custom model:")
        print(set(sd_keys_hf) - set(sd_keys))

        print("Keys in the custom model but not in the HuggingFace model:")
        print(set(sd_keys) - set(sd_keys_hf))

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            elif k in sd_keys:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_non_decay_params = sum(p.numel() for p in non_decay_params)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f" Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# Define the Combined Model
class CombinedModel(nn.Module):
    def __init__(self, gpt_config, input_dim, expert_dim, num_experts, model_dim, num_timesteps, num_output_heads):
        super(CombinedModel, self).__init__()
        self.gpt = GPT(gpt_config)
        self.moelayer = MoELayer(num_experts, expert_dim, model_dim)
        self.diffusion_model = DiffusionModel(model_dim, num_timesteps)
        self.vae = VAE(input_dim, expert_dim, model_dim)
        self.mdn_rnn = MDNRNN(model_dim, expert_dim, model_dim, num_gaussians=5)
        self.output_heads = nn.ModuleList([nn.Linear(model_dim, input_dim) for _ in range(num_output_heads)])

    def forward(self, idx, targets=None):
        # GPT Forward
        gpt_logits, gpt_loss = self.gpt(idx, targets)

        # MoE Layer
        moe_output = self.moelayer(gpt_logits)

        # Diffusion Model
        t = torch.randint(0, self.diffusion_model.timesteps, (moe_output.size(0),)).to(moe_output.device)
        diffusion_output = self.diffusion_model(moe_output, t)

        # VAE Forward
        recon_x, mu, logvar = self.vae(diffusion_output)

        # MDN-RNN Forward
        h = (torch.zeros(1, self.mdn_rnn.hidden_dim).to(diffusion_output.device), torch.zeros(1, self.mdn_rnn.hidden_dim).to(diffusion_output.device))
        rnn_output, _ = self.mdn_rnn(diffusion_output, h)

        # Multi-token prediction
        multi_token_outputs = [head(rnn_output) for head in self.output_heads]

        return multi_token_outputs, gpt_loss, recon_x, mu, logvar

    def compute_loss(self, idx, targets):
        multi_token_outputs, gpt_loss, recon_x, mu, logvar = self.forward(idx, targets)

        # Compute VAE loss
        vae_loss = self.vae.loss_function(recon_x, idx, mu, logvar)

        # Compute MDN-RNN loss
        mean, log_sigma, log_pi = self.mdn_rnn.get_mixture_coef(multi_token_outputs[0])
        mdn_rnn_loss = self.mdn_rnn.loss_function(idx, mean, log_sigma, log_pi)

        # Total loss
        total_loss = gpt_loss + vae_loss + mdn_rnn_loss

        return total_loss

# Custom Dataset Class for Loading Numpy Files
class NpyDataset(Dataset):
    def __init__(self, npy_files):
        self.data = []
        for npy_file in npy_files:
            loaded_data = np.load(npy_file, allow_pickle=True)
            self.data.extend(loaded_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Ensure the item is a tensor and has the correct shape
        return torch.tensor(item, dtype=torch.long)

# Training Loop
def train_model(model, dataloader, optimizers, num_epochs, device, mcts_config):
    model.train()
    for epoch in range(num_epochs):
        for idx in dataloader:
            idx = idx.to(device)
            print(f"Data shape: {idx.shape}")  # Add this line to check data shape

            # Zero the parameter gradients for all optimizers
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            # Compute loss and perform backpropagation
            loss = model.compute_loss(idx, idx)
            loss.backward()

            # Step all optimizers
            for optimizer in optimizers.values():
                optimizer.step()

            logger.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

        # Validation with MCTS
        validate_with_mcts(model, val_dataloader, device, mcts_config)


        # Validation with MCTS
        validate_with_mcts(model, val_dataloader, device, mcts_config)

def validate_with_mcts(model, dataloader, device, mcts_config):
    model.eval()
    for idx in dataloader:
        idx = idx.to(device)
        prompt_text = tokenizer.decode(idx[0], skip_special_tokens=True)
        root = Node(prompt=prompt_text, text=prompt_text)

        # Perform MCTS iterations
        for _ in range(mcts_config['num_iterations']):
            node = root
            while not node.is_leaf():
                node = select_node(node)
            expand_node(node, tokenizer, model, model.vae, model.mdn_rnn, mcts_config['num_expansions'])
            reward = simulate(node, model, st_model, tokenizer, [relevance_reward, coherence_reward], model.vae, model.mdn_rnn)
            backpropagate(node, reward)

        # Get the best continuation
        best_child = max(root.children, key=lambda x: x.reward)
        logger.info("Best continuation: %s", best_child.text)

# Hyperparameters
input_dim = 768
expert_dim = 1024
num_experts = 4
model_dim = 768
num_timesteps = 1000
num_output_heads = 4
num_layers = 12
n_head = 8
num_epochs = 10
learning_rate = 1e-4

# MCTS configuration
mcts_config = {
    'num_iterations': 10,
    'num_expansions': 5
}

# Model and Configurations
gpt_config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=num_layers, n_head=n_head, n_embd=model_dim)
model = CombinedModel(gpt_config, input_dim, expert_dim, num_experts, model_dim, num_timesteps, num_output_heads).to(device)

# Separate optimizers for each model component
optimizers = {
    'gpt': torch.optim.Adam(model.gpt.parameters(), lr=learning_rate),
    'moe': torch.optim.Adam(model.moelayer.parameters(), lr=learning_rate),
    'diffusion': torch.optim.Adam(model.diffusion_model.parameters(), lr=learning_rate),
    'vae': torch.optim.Adam(model.vae.parameters(), lr=learning_rate),
    'mdn_rnn': torch.optim.Adam(model.mdn_rnn.parameters(), lr=learning_rate)
}

# Load the dataset from npy files
train_files = [f"C:/Users/Admin/DATAS/edu_fineweb10B/edufineweb_train_{i:06d}.npy" for i in range(1, 10)]
val_file = "C:/Users/Admin/DATAS/edu_fineweb10B/edufineweb_val_000000.npy"

train_dataset = NpyDataset(train_files)
val_dataset = NpyDataset([val_file])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Initialize and load the models
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
st_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Train the model
train_model(model, train_dataloader, optimizers, num_epochs, device, mcts_config)

logger.info("Training complete!")
