
import time

import numpy as np
import torch


from torch.optim import Adam
from torch.utils.data import DataLoader

from deepspeed import deepspeed
import torch.nn as nn

import ipc
from config import build_parser
from model import Transformer
from data_loader import Dataset_ETT_hour, Dataset_ETT_minute
from metrics import metric


def get_model(args):
    return Transformer(args.embedding_size, args.hidden_size, args.input_len, args.dec_seq_len, args.pred_len,
                       output_len=args.output_len,
                       n_heads=args.n_heads, n_encoder_layers=args.n_encoder_layers,
                       n_decoder_layers=args.n_decoder_layers, enc_attn_type=args.encoder_attention,
                       dec_attn_type=args.decoder_attention, dropout=args.dropout)


def get_params(mdl):
    return mdl.parameters()


def _get_data(args, flag):
    if not args.data == 'ETTm1':
        Data = Dataset_ETT_hour
    else:
        Data = Dataset_ETT_minute
    # timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False;
        drop_last = True;
        batch_size = 32
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False;
        drop_last = False;
        batch_size = 1;
        # freq = args.detail_freq
        # Data = Dataset_Pred
    else:
        shuffle_flag = True;
        drop_last = True;
        batch_size = args.batch_size
        # freq = args.freq

    data_set = Data(
        root_path='data',
        data_path=args.data+'.csv',
        flag=flag,
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        # timeenc=timeenc,
        # freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader


def run_metrics(caption, preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)
    # print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # print('test shape:', preds.shape, trues.shape)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('{} ; MSE: {}, MAE: {}'.format(caption, mse, mae))
    return mse, mae


def run_iteration(model, loader, args, training=True, message = ''):
    preds = []
    trues = []
    total_loss = 0
    elem_num = 0
    steps = 0
    target_device = 'cuda:{}'.format(args.local_rank)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        if not args.deepspeed:
            model.optim.zero_grad()

        batch = torch.tensor(batch_x, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        target = torch.tensor(batch_y, dtype=torch.float16 if args.fp16 else torch.float32,
                              device=target_device)

        elem_num += len(batch)
        steps += 1

        result = model(batch)

        loss = nn.functional.mse_loss(result.squeeze(2), target.squeeze(2), reduction='mean')

        #pred = result.detach().cpu().unsqueeze(2).numpy()  # .squeeze()
        pred = result.detach().cpu().numpy()  # .squeeze()
        true = target.detach().cpu().numpy()  # .squeeze()

        preds.append(pred)
        trues.append(true)

        unscaled_loss = loss.item()
        total_loss += unscaled_loss
        print("{} Loss at step {}: {}, mean for epoch: {}, mem_alloc: {}".format(message, steps, unscaled_loss, total_loss / steps,torch.cuda.max_memory_allocated()))

        if training:
            if args.deepspeed:
                model.backward(loss)
                model.step()
            else:
                loss.backward()
                model.optim.step()
    return preds, trues


def preform_experiment(args):
    model = get_model(args)
    params = list(get_params(model))
    print('Number of parameters: {}'.format(len(params)))
    for p in params:
        print(p.shape)

    if args.deepspeed:
        deepspeed_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                              model=model,
                                                              model_parameters=params)
    else:
        model.to('cuda')
        model.optim = Adam(params, lr=0.001)

    train_data, train_loader = _get_data(args, flag='train')



    start = time.time()
    for iter in range(1, args.iterations + 1):
        preds, trues = run_iteration(deepspeed_engine if args.deepspeed else model , train_loader, args, training=True, message=' Run {:>3}, iteration: {:>3}:  '.format(args.run_num, iter))
        mse, mae = run_metrics("Loss after iteration {}".format(iter), preds, trues)
        if args.local_rank == 0:
            ipc.sendPartials(iter, mse, mae)
        print("Time per iteration {}, memory {}".format((time.time() - start)/iter, torch.cuda.memory_stats()))

    print(torch.cuda.max_memory_allocated())

    if args.debug:
        model.record()


    test_data, test_loader = _get_data(args, flag='test')
    if deepspeed:
        model.inference()
    else:
        model.eval()
    # Model evaluation on validation data
    v_preds, v_trues = run_iteration(deepspeed_engine if args.deepspeed else model, test_loader, args, training=False, message="Validation set")
    mse, mae = run_metrics("Loss for validation set ", v_preds, v_trues)

    # Send results / plot models if debug option is on
    if args.local_rank == 0:
        ipc.sendResults(mse, mae)
        if args.debug:
            plot_model(args, model)

def main():
    parser = build_parser()
    args = parser.parse_args(None)
    preform_experiment(args)


if __name__ == '__main__':
    main()


