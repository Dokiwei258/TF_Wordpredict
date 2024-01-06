from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split = 'train')

tokenizer = get_tokenizer('basic_english')

vocab = build_vocab_from_itertor(map(tokenizer,train_iter), specials = ['<unk>'])

vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
  data = [torch.tensor(vocab(tokenizer(item)),dtype = torch.long) for item in raw_text_iter]

  return torch.cat(tuple(filter(lambda t: t.numel() > 0,data)))

train_iter, val_iter, test_iter = WikiText2()

train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data: Tensor, bsz: int) -> Tensor:
  seq_len = data.size(0) // bsz 
  data = data[:seq_len * bsz]
  data = data.view(bdz,seq_len).t().contiguous()
  return data.to(device)

batch_size = 20
eval_batch_size = 10

train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 35

def get_batch(source: Tensor, i:int) -> Tuple[Tensor, Tensor]:
  seq_len = min(bptt, len(source) - 1 - i)
  data = source[i:i_seq_len]
  target = source[i+1:i+1+seq_len].reshape(-1)

  return data, target


