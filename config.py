import argparse
import sys
import json

from deepspeed import deepspeed


class Config:
    def __init__(self, data='ETTh1',seq_len=720, pred_len=24, dec_seq_len=24, hidden_size=128, heads=3, batch_size=100, embedding_size=32,
                 n_encoder_layers=3, encoder_attention='full', n_decoder_layers=3, decoder_attention='full',
                 prediction_type='uni', dropout=0.1, fp16=True,
                 iterations=10, exps=5, deepspeed= True, debug=False):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dec_seq_len = dec_seq_len
        self.hidden_size = hidden_size
        self.heads = heads
        self.n_encoder_layers = n_encoder_layers
        self.encoder_attention = encoder_attention
        self.n_decoder_layers = n_decoder_layers
        self.decoder_attention = decoder_attention
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.prediction_type = prediction_type
        self.dropout = dropout
        self.fp16 = fp16
        self.deepspeed = deepspeed
        self.iterations = iterations
        self.exps = exps
        self.debug = debug

    def extend_argv(self):
        sys.argv.extend(['--data', str(self.data)])
        sys.argv.extend(['--seq_len', str(self.seq_len)])
        sys.argv.extend(['--pred_len', str(self.pred_len)])
        sys.argv.extend(['--dec_seq_len', str(self.dec_seq_len)])
        sys.argv.extend(["--hidden_size", str(self.hidden_size)])
        sys.argv.extend(["--n_encoder_layers", str(self.n_encoder_layers)])
        sys.argv.extend(["--n_decoder_layers", str(self.n_decoder_layers)])
        sys.argv.extend(["--encoder_attention", str(self.encoder_attention)])
        sys.argv.extend(["--decoder_attention", str(self.decoder_attention)])
        sys.argv.extend(["--n_heads", str(self.heads)])
        sys.argv.extend(["--batch_size", str(self.batch_size)])
        sys.argv.extend(["--embedding_size", str(self.embedding_size)])
        sys.argv.extend(["--iterations", str(self.iterations)])
        sys.argv.extend(["--exps", str(self.exps)])

        sys.argv.extend(["--dropout", str(self.dropout)])
        if self.fp16:
            sys.argv.extend(["--fp16"])

        if self.deepspeed:
            sys.argv.extend(["--deepspeed"])

        if self.debug:
            sys.argv.extend(["--debug"])

        if self.prediction_type == 'uni':
            sys.argv.extend(["--features", 'S'])
            sys.argv.extend(["--input_len", '1', "--output_len", "1"])
        elif self.prediction_type == 'multi':
            sys.argv.extend(["--features", 'M'])
            sys.argv.extend(["--input_len", '7', "--output_len", "7"])

        else:
            raise NotImplemented

    def __str__(self):
        res = '::  ds-time-series config\n'
        res += ':::: train dataset: {}\n'.format(self.data)
        res += ':::: train input sequence len: {}\n'.format(self.seq_len)
        res += ':::: train prediction sequence len: {}\n'.format(self.pred_len)
        res += ':::: train decoder sequence len: {}\n'.format(self.dec_seq_len)
        res += ':::: train batch size: {}\n'.format(self.batch_size)
        res += ':::: train prediction type: {}\n'.format('univariate' if self.prediction_type == 'uni'
                                                         else 'multiunivariate')
        res += ':::: train iterations: {}\n'.format(str(self.iterations))
        res += ':::: train experiment number: {}\n'.format(self.exps)
        res += ':::: train using deepspeed: {}\n'.format(self.deepspeed)
        res += ':::: train using fp16: {}\n'.format(self.deepspeed)
        res += ':::: train recording: {}\n'.format(self.debug)

        res += ':::: model hidden size: {}\n'.format(self.hidden_size)
        res += ':::: model embedding size: {}\n'.format(self.embedding_size)
        res += ':::: model encoder layers: {}\n'.format(self.n_encoder_layers)
        res += ':::: model encoder attention: {}\n'.format(self.encoder_attention)
        res += ':::: model decoder layers: {}\n'.format(self.n_encoder_layers)
        res += ':::: model decoder attention: {}\n'.format(self.decoder_attention)
        res += ':::: model heads number: {}\n'.format(self.heads)
        res += ':::: model input dropout: {}\n'.format(self.dropout)

        return res

    @staticmethod
    def from_file(f):
        with open(f, 'r') as file:
            a = file.readlines()
            dict = json.loads(''.join(a))
        return Config(**dict)

    def to_json(self):
        return json.dumps(self.__dict__, indent=2)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['ETTh1', 'ETTh2', 'ETTm1'], required=True)
    parser.add_argument('--input_len', type=int, required=True)
    parser.add_argument('--output_len', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--dec_seq_len', type=int, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--target', default='OT', type=str)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--exps', type=int, required=True)

    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--n_heads', type=int, required=True)
    parser.add_argument('--n_encoder_layers', type=int, required=True)
    parser.add_argument('--encoder_attention', type=str, required=True)
    parser.add_argument('--n_decoder_layers', type=int, required=True)
    parser.add_argument('--decoder_attention', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--embedding_size', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--num-workers',
                        type=int,
                        default=2)
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--run_num", type=int, default=0)
    parser.add_argument('--debug', action='store_true')

    parser = deepspeed.add_config_arguments(parser)
    return parser