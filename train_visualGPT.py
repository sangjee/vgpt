import imp
import random

from CustomDataModule import CustomDataModule, build_loaders

from data import ImageDetectionsField, TextField
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer_visualgpt, VisualEncoder, ScaledDotProductAttention
import torch
from torch.optim import Adam
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import logging

from transformers import AdamW


from models.captioning_model import CaptioningModel

import pandas as pd
import os

def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, i in enumerate(dataloader):


                detections, captions = i['image'].to(device), i['text'].to(device)
                out,past = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, exp_name, epoch):
    import itertools

    model.eval()

    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, i in enumerate(iter(dataloader)):

            images = i['image'].to(device)
            caps_gt = i['text']

            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<|endoftext|>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
  

                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = [gts_i]
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


def train_xe(model, dataloader, text_field,gpt_optimizer,dataloader_eval,args):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, i in enumerate(dataloader):
            detections, captions = i['image'].to(device), i['text'].to(device)


            out,past= model(detections, captions)

            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()

            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            gpt_optimizer.step()
            gpt_optimizer.zero_grad()


            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, cider, text_field,gpt_optimizer,args):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool(processes=1)
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 1
    out_size = 1

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, i in enumerate(dataloader):
            caps_gt = i['text']
            detections = i['image'].to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<|endoftext|>'],
                                                beam_size, out_size=out_size)

            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * out_size for c in caps_gt)))

            # caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            caps_gen = evaluation.PTBTokenizer.tokenize(caps_gen)
            caps_gt = evaluation.PTBTokenizer.tokenize(caps_gt)
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)

            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], out_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()

            loss.backward()

            if (it + 1) % args.gradient_accumulation_steps == 0 or (it+1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                gpt_optimizer.step()
                gpt_optimizer.zero_grad()


            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()



    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VisualGPT')
    parser.add_argument('--exp_name', type=str, default='visualGPT')
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--val_data_path', type=str)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--head', type=int, default=12)
    parser.add_argument('--logs_folder', type=str, default='/home/lab/sangjee/strok/tensorlog')
    parser.add_argument('--random_seed', type = int, default="42")
    parser.add_argument('--gpt_model_type',type=str, default= "gpt")
    parser.add_argument('--lr', type = float, default=1e-4)
    parser.add_argument('--log_file',type = str, default="/home/lab/sangjee/strok/log/visualGPT.txt")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")


    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--optimizer_type', type= str, default = "adamw")
    parser.add_argument('--max_grad_norm', default=1.0, type = float)
    parser.add_argument('--train_percentage', default=1.0, type = float)
    parser.add_argument('--reinforcement_lr',type = float, default=1e-5)
    parser.add_argument("--decoder_layer", type= int, default = 12)
    parser.add_argument("--encoder_layer",type=int, default=3)
    parser.add_argument("--tau",type=float, default = 0.0)
    parser.add_argument("--data_type", type= str, default = "mri")
    parser.add_argument("--data_channel",type=int, default = 3)

    args = parser.parse_args()

    resume_last = True
    resume_best = True


    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    


    os.environ["TOKENIZERS_PARALLELISM"] = "True"

    n_gpus = torch.cuda.device_count()

    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    logging.info(args)
    #
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=None, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<?', eos_token='<|endoftext|>', fix_length=55, lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    data_type = args.data_type

    train_path = args.train_data_path
    test_path = args.test_data_path
    val_path = args.val_data_path
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)

    if data_type == 'mri':
        train_df.rename(columns={'image':'image_nii','image_hdf5':'image'},inplace=True)
        test_df.rename(columns={'image':'image_nii','image_hdf5':'image'},inplace=True)
        val_df.rename(columns={'image':'image_nii','image_hdf5':'image'},inplace=True)


    data_module = CustomDataModule(train_df=train_df, val_df=val_df, test_df=test_df, batch_size=args.batch_size, num_workers=args.num_workers, tokenizer=text_field, mode='train', d_type=data_type, channel=args.data_channel)
    data_module.prepare_data()
    data_module.setup()

    
    data_module2 = CustomDataModule(train_df=train_df, val_df=val_df, test_df=test_df, batch_size=args.batch_size, num_workers=args.num_workers, tokenizer=text_field, mode='valid', d_type=data_type, channel=args.data_channel)
    data_module2.prepare_data()
    data_module2.setup()


    # Create the dataset
    # dataset = COCO(image_field, text_field, train_df, test_df, val_df)
    # train_dataset, val_dataset, test_dataset = dataset.splits

    train_dataset = build_loaders(train_df, text_field, mode='train', d_type=data_type, channel=args.data_channel)

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_GPT_vocab("data/encoder.json")
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))


    # Model and dataloaders

    encoder = VisualEncoder(args.encoder_layer, 0, channel=args.data_channel, attention_module=ScaledDotProductAttention)
    model = Transformer_visualgpt(text_field.vocab.stoi['<?'], encoder, args.gpt_model_type, args.decoder_layer,tau=args.tau).to(device)
    
    dict_dataset_train = build_loaders(train_df, text_field, mode='valid', d_type=data_type, channel=args.data_channel)
    # ref_caps_train = []
    # for i in dict_dataset_train:
    #     ref_caps_train.append(i['text'])
    # cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))

    total_step_number = int(len(train_dataset)/(args.batch_size * args.gradient_accumulation_steps)*100)
 

    if args.optimizer_type =="adamw":
        
        gpt_optimizer = AdamW(model.parameters(),lr=args.lr,betas=(0.9, 0.999), eps=1e-8)
  
    elif args.optimizer_type =="adam":
        optimizer = Adam(model.parameters(), lr = args.lr)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['+='])
    use_rl = False
    best_cider = .0
    patience = 0
    start_epoch = 0

    if resume_last or resume_best:
        if resume_last:
            fname = '/home/lab/sangjee/strok/saved_models/%s_last.pth' % args.exp_name
        else:
            fname = '/home/lab/sangjee/strok/saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            gpt_optimizer.load_state_dict(data['optimizer'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    # use_rl=True
    for e in range(start_epoch, start_epoch + args.epoch):
        dataloader_train = data_module.train_dataloader()
        dataloader_val = data_module.val_dataloader()
        dict_dataloader_train = data_module2.train_dataloader()
        dict_dataloader_val = data_module2.val_dataloader()
        dict_dataloader_test = data_module2.test_dataloader()

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, text_field,gpt_optimizer,dataloader_val,args)

            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            print('----------------')
     
            # train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, cider_train, text_field,
            #                                                      gpt_optimizer, args)
            # writer.add_scalar('data/train_loss', train_loss, e)
            # writer.add_scalar('data/reward', reward, e)
            # writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, args.exp_name+"_val", str(e))
        # val_cider = scores['CIDEr']
        val_cider = scores['BLEU'][0]
        writer.add_scalar('data/val_cider', scores['CIDEr'], e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        # writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        print('val_cider : ',val_cider,'val_BLUE1 : ',scores['BLEU'][0],'val_BLUE4 : ',scores['BLEU'][3])

        logging.info("val cider"+str(val_cider)+"current epoch "+str(e))
        logging.info("val bleu1" + str(scores["BLEU"][0]) + "current epoch " + str(e))
        logging.info("val bleu4" + str(scores["BLEU"][3]) + "current epoch " + str(e))
        # logging.info("val meteor"+str(scores["METEOR"])+"current epoch "+str(e))
        logging.info("val rouge" + str(scores["ROUGE"]) + "current epoch " + str(e))



        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field, args.exp_name+"_test", str(e))
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        # writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        logging.info("test cider" + str(scores['CIDEr']) + "current epoch " + str(e))
        logging.info("test bleu1" + str(scores["BLEU"][0]) + "current epoch " + str(e))
        logging.info("test bleu4" + str(scores["BLEU"][3]) + "current epoch " + str(e))
        # logging.info("test meteor" + str(scores["METEOR"]) + "current epoch " + str(e))
        logging.info("test rouge" + str(scores["ROUGE"]) + "current epoch " + str(e))

      
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False


        if patience == args.patience:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0

                gpt_optimizer = AdamW(model.parameters(),
                                     lr = args.reinforcement_lr,betas=(0.9, 0.999), eps=1e-8)

                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            print(" now we are resuming!!!!")
            data = torch.load('/home/lab/sangjee/strok/saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': gpt_optimizer.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, '/home/lab/sangjee/strok/saved_models/%s_last.pth' % args.exp_name)

        if best:
            copyfile('/home/lab/sangjee/strok/saved_models/%s_last.pth' % args.exp_name, '/home/lab/sangjee/strok/saved_models/%s_best.pth' % args.exp_name)

