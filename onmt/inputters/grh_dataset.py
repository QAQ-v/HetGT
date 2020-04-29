import six
import json
import torch
import numpy as np
from functools import partial
from torchtext.data import  RawField, Pipeline
from torchtext.data.utils import get_tokenizer

from onmt.inputters.datareader_base import DataReaderBase


class GrhDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read edges data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to edge file or iterable of the actual edge data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` , ``"tgt" or "grh``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with GrhDataReader."
        # assert _dir is not None or _dir != "", \
        #     "Must use _dir with GrhDataReader (provide edges vocab)."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        # vocab = json.load(_dir)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}

class GraphField(RawField):
    """ custom field.

    Because the grh data doesnt need to do numericalization (what the default field always do
    in process()) and pad (can set sequential=False to avoid it, but you cannot do the tokenize
    either). So we need to customize the special field which does our wanted operation.

    Notice that here we dont implement multi-shards.
    """
    def __init__(self, sequential=True, use_vocab=True, preprocessing=None,
                 # 等价于 tok=lambda s: s.split(), tokenize=tok 还是默认参数的用法
                 postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 dtype=torch.long
                 ):

        super(GraphField, self).__init__()
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.dtype = dtype

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types) and
                not isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        # The Pipeline that will be applied to examples using this field after
        # tokenizing but before numericalizing. Many Datasets replace this
        # attribute with a custom preprocessor. Default: None.
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
            Graph information is in the form of an adjacency list.
            We convert this to an adjacency matrix in NumPy format.
            The matrix contains label ids.
            IMPORTANT: we add one to the label id stored in the matrix.
            This is because 0 is a valid vocab id but we want to use 0's
            to represent lack of edges instead. This means that the GCN code
            should account for this.
            But it is better to be defined in postprocessing

        Args:
            batch (list(object)): A list of object from a batch of examples.
        """
        def get_pad_len(l):
            last_tuple = l[-1]
            last_node, edge_type = last_tuple[0], last_tuple[-1]
            #assert edge_type == 3, "must be the self-connection"
            # assert last_node == last_tuple[1]
            return last_node
        # global_index_list = []
        # sorted(final_grh, key=lambda x: int(x[0]))
        pad_len = max(list(map(get_pad_len, batch)))+1
        new_grh = np.array([np.zeros((pad_len, pad_len)) for _ in range(len(batch))])
        for i, grh in enumerate(batch):
            # global_index = 0
            for tup in grh:
                new_grh[i][tup[0]][tup[1]] = tup[2] + 1
                if tup[0] == tup[1]:
                    self_id = tup[2]
            #     if tup[0] > global_index:
            #         global_index = tup[0]
            # global_index_list.append(global_index)
            for j in range(pad_len):
                new_grh[i][j][j] = self_id + 1# the pad symbols need to have a self loop (还是直接mask掉?)
        # error1. 因为field是preprocess就构造然后存到dataset里的, 所以加载时不会再执行init
        # error2. torch.Tensor 是legacy constructor不要再用了! 它不支持device=str, 用torch.tensor!
        arr = torch.tensor(new_grh,device=device, dtype=self.dtype)
        if self.sequential:
                arr = arr.contiguous()
        return arr


def _edge_tokenizer(string, vocab=None):
    assert vocab is not None, "the edges vocab cannot be None"
    graph_tokens = string.rstrip().split()
    adj_list = [(int(tok[1:-1].split(',')[0]),
                 int(tok[1:-1].split(',')[1]),
                 vocab[tok[1:-1].split(',')[2]]) for tok in graph_tokens]
    return adj_list

# def _convert_to_adj_matrix():


def grh_fields(**kwargs):
    """Create graph fields."""
    vocab_dir = kwargs.get("vocab")
    with open(vocab_dir, encoding='utf8') as inp:
        vocab = json.load(inp)
        tokenize = partial(
            _edge_tokenizer,
            vocab=vocab
        )
        feat = GraphField(
            sequential=False,
            use_vocab=False,
            tokenize=tokenize, )
        return feat