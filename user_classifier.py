import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datasets
from absl import app, flags
import wandb
import os
from transformers import get_cosine_schedule_with_warmup
from collections import defaultdict
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_integer('epochs', 10, 'Number of epochs for training')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay for training')
flags.DEFINE_integer('input_size', 3584, 'Input size of the model') 
flags.DEFINE_integer('hidden_size', 512, 'Hidden size of the model')
flags.DEFINE_integer('num_layers', 4, 'Number of layers in the model')
flags.DEFINE_integer('num_classes', 304, 'Number of unique users')
flags.DEFINE_string('dataset_name', 'Asap7772/emb_classify', 'Name of the model')
flags.DEFINE_string('wandb_project', 'user-classifier-rerun-0907', 'Wandb project name')
flags.DEFINE_float('warmup_ratio', 0.1, 'Warmup ratio for the scheduler')
flags.DEFINE_float('dropout', 0.1, 'Dropout probability')
flags.DEFINE_integer('num_test_batches', 10, 'Number of test batches to evaluate')
flags.DEFINE_string('output_dir', '/home/anikait.singh/personalized-t2i/checkpoints/user_classifier', 'Output directory to save the model')

def get_accuracy(probs, class_, k=1):
    if k == 1:
        # Top-1 accuracy (standard classification accuracy)
        pred = torch.argmax(probs, dim=-1)
        correct = torch.sum(pred == class_)
        return correct.item() / class_.size(0)
    else:
        # Top-k accuracy
        top_k = torch.topk(probs, k, dim=-1).indices
        correct = torch.sum(top_k.eq(class_.unsqueeze(-1)))
        return correct.item() / class_.size(0)

class UserClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, drop_p=0.1):
        super(UserClassifier, self).__init__()
        
        # Define the model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.activation_layer = nn.GELU()
        
        self.mlp_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.mlp_activation_layers = nn.ModuleList([nn.GELU() for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(drop_p) for _ in range(num_layers)])
        
        self.output_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_layer(x)
        
        for i in range(self.num_layers):
            x = x + self.mlp_layers[i](x)
            x = self.mlp_activation_layers[i](x)
            x = self.norm_layers[i](x)
            x = self.dropout_layers[i](x)
        
        x = self.output_layer(x)    
        
        return x
    
def main(_):
    random_str = np.random.bytes(4).hex()
    unique_run_name = f'{FLAGS.dataset_name}-{FLAGS.batch_size}-{FLAGS.epochs}-{FLAGS.learning_rate}-{FLAGS.weight_decay}-{FLAGS.input_size}-{FLAGS.hidden_size}-{FLAGS.num_layers}'
    unique_run_name = unique_run_name.replace('/', '-').replace(' ', '_').replace('.', '_')
    unique_run_name = f'{unique_run_name}-{random_str}'
    
    ds = datasets.load_dataset(FLAGS.dataset_name)
    remove_cols = list(ds['train'].column_names)
    remove_cols.remove('class')
    remove_cols.remove('emb')
    def map_fn(examples):
        for i in range(len(examples['emb'])):
            examples['emb'][i] = examples['emb'][i][-1]
        return examples
    ds = ds.map(map_fn, batched=True, num_proc=os.cpu_count(), remove_columns=remove_cols)
    
    train_dataloader = torch.utils.data.DataLoader(
        ds['train'], 
        batch_size=FLAGS.batch_size, 
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        ds['test'],
        batch_size=FLAGS.batch_size, 
        shuffle=True,
    )
    
    model = UserClassifier(
        input_size=FLAGS.input_size,
        hidden_size=FLAGS.hidden_size,
        num_classes=FLAGS.num_classes,
        num_layers=FLAGS.num_layers
    )
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    num_steps = len(train_dataloader) * FLAGS.epochs
    warmup_steps = int(FLAGS.warmup_ratio * len(train_dataloader))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)
    
    config_dict = FLAGS.flag_values_dict()
    wandb.init(project=FLAGS.wandb_project, config=config_dict)
    
    def process_batch(data):
        emb = torch.stack(data['emb']).T.to(device='cuda', dtype=torch.float)
        class_ = data['class'].to(device='cuda', dtype=torch.long)
        return emb, class_
    
    curr_step = 0
    for epoch in range(FLAGS.epochs):
        for i, data in enumerate(train_dataloader):
            exact_epoch = epoch + i / len(train_dataloader)
            emb, class_ = process_batch(data)
            
            optimizer.zero_grad()
            logits = model(emb)
            probs = F.softmax(logits, dim=-1)
            
            loss = criterion(logits, class_)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_stats = dict()
            
            train_stats['loss'] = loss.item()
            train_stats['accuracy'] = get_accuracy(probs, class_)
            train_stats['top2_accuracy'] = get_accuracy(probs, class_, k=2)
            train_stats['top4_accuracy'] = get_accuracy(probs, class_, k=4)
            train_stats['top8_accuracy'] = get_accuracy(probs, class_, k=8)
            train_stats['top16_accuracy'] = get_accuracy(probs, class_, k=16)
            train_stats['top32_accuracy'] = get_accuracy(probs, class_, k=32)
            train_stats['epoch'] = exact_epoch
            train_stats['lr'] = optimizer.param_groups[0]['lr']
            
            desired_prob = probs[torch.arange(probs.size(0)), class_]
            train_stats['desired_prob'] = desired_prob.mean().item()
            
            train_stats = {f'train/{key}': value for key, value in train_stats.items()}
            wandb.log(train_stats, step=curr_step)
            
            if i % 100 == 0:
                with torch.no_grad():
                    test_stats = defaultdict(float)
                    for j, test_data in enumerate(test_dataloader):
                        emb, class_ = process_batch(test_data)
                        
                        logits = model(emb)
                        probs = F.softmax(logits, dim=-1)
                        
                        loss = criterion(logits, class_)
                        
                        desired_prob = probs[torch.arange(probs.size(0)), class_]
                        accuracy = get_accuracy(probs, class_)
                        
                        test_stats['loss'] += loss.item()
                        test_stats['accuracy'] += accuracy
                        test_stats['top2_accuracy'] += get_accuracy(probs, class_, k=2)
                        test_stats['top4_accuracy'] += get_accuracy(probs, class_, k=4)
                        test_stats['top8_accuracy'] += get_accuracy(probs, class_, k=8)
                        test_stats['top16_accuracy'] += get_accuracy(probs, class_, k=16)
                        test_stats['top32_accuracy'] += get_accuracy(probs, class_, k=32)
                        test_stats['desired_prob'] += desired_prob.mean().item()
                        test_stats['num_batches'] += 1
                        
                        if test_stats['num_batches'] == FLAGS.num_test_batches:
                            break
                    
                    for key in test_stats:
                        if key != 'num_batches':
                            test_stats[key] /= test_stats['num_batches']
                    log_dict = {f'test/{key}': value for key, value in test_stats.items()}  
                    print(f'Epoch: {exact_epoch}, Step: {i}, Loss: {loss.item()}, Accuracy: {accuracy}, Test Loss: {test_stats["loss"]}, Test Accuracy: {test_stats["accuracy"]}')
                    wandb.log(log_dict, step=curr_step)
            else:
                print(f'Epoch: {exact_epoch}, Step: {i}, Loss: {loss.item()}, Accuracy: {accuracy}')
            curr_step += 1
        
        # Save model
        output_dir = os.path.join(FLAGS.output_dir, unique_run_name, f'epoch_{epoch}')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pth'))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pth'))
        
    wandb.finish()
    
    

if __name__ == '__main__':
    app.run(main)