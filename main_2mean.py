import argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import random

from data_loader import load_data
from model import *
from utils import *

EOS = 1e-10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def train_cl(cl_model, discriminator0, discriminator1, optimizer_cl, features, str_encodings, edges):

    cl_model.train()
    discriminator0.eval()
    discriminator1.eval()

    adj_1_0, adj_2_0, weights_lp, _ = discriminator0(torch.cat((features, str_encodings), 1), edges)
    adj_1_1, adj_2_1, weights_lp, _ = discriminator1(torch.cat((features, str_encodings), 1), edges)

    adj_1 = (adj_1_0 + adj_1_1)/2
    adj_2 = (adj_2_0 + adj_2_1)/2

    features_1, adj_1, features_2, adj_2 = augmentation(features, adj_1, features, adj_2, args, cl_model.training)
    cl_loss = cl_model(features_1, adj_1, features_2, adj_2)

    optimizer_cl.zero_grad()
    cl_loss.backward()
    optimizer_cl.step()

    return cl_loss.item()


def train_discriminator(cl_model, discriminator0, optimizer_disc0, discriminator1, optimizer_disc1, features, str_encodings, edges, args):

    cl_model.eval()
    discriminator0.train()
    discriminator1.train()

    adj_1_0, adj_2_0, weights_lp_0, weights_hp_0 = discriminator0(torch.cat((features, str_encodings), 1), edges)
    adj_1_1, adj_2_1, weights_lp_1, weights_hp_1 = discriminator1(torch.cat((features, str_encodings), 1), edges)

    adj_1 = (adj_1_0 + adj_1_1)/2
    adj_2 = (adj_2_0 + adj_2_1)/2
    weights_hp = (weights_hp_0 + weights_hp_1)/2
    weights_lp = (weights_lp_0 + weights_lp_1)/2

    rand_np = generate_random_node_pairs(features.shape[0], edges.shape[1])
    psu_label = torch.ones(edges.shape[1]).cuda()

    embedding = cl_model.get_embedding(features, adj_1, adj_2)
    edge_emb_sim = F.cosine_similarity(embedding[edges[0]], embedding[edges[1]])

    rnp_emb_sim_lp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_lp = F.margin_ranking_loss(edge_emb_sim, rnp_emb_sim_lp, psu_label, margin=args.margin_hom, reduction='none')
    loss_lp *= torch.relu(weights_lp - 0.5)

    rnp_emb_sim_hp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_hp = F.margin_ranking_loss(rnp_emb_sim_hp, edge_emb_sim, psu_label, margin=args.margin_het, reduction='none')
    loss_hp *= torch.relu(weights_hp - 0.5)

    rank_loss = (loss_lp.mean() + loss_hp.mean()) / 2

    optimizer_disc0.zero_grad()
    optimizer_disc1.zero_grad()
    rank_loss.backward()
    optimizer_disc0.step()
    optimizer_disc1.step()


    return rank_loss.item()


def main(args):

    setup_seed(0)
    features, edges, str_encodings, train_mask, val_mask, test_mask, labels, nnodes, nfeats = load_data(args.dataset)
    results = []

    for trial in range(args.ntrials):
        print(">>>>>> trial: ", trial, " <<<<<<")
        setup_seed(trial)

        cl_model = GCL(nlayers=args.nlayers_enc, nlayers_proj=args.nlayers_proj, in_dim=nfeats, emb_dim=args.emb_dim,
                    proj_dim=args.proj_dim, dropout=args.dropout, sparse=args.sparse, batch_size=args.cl_batch_size).cuda()
        cl_model.set_mask_knn(features.cpu(), k=args.k, dataset=args.dataset)
        discriminator0 = Edge_Discriminator(nnodes, nfeats + str_encodings.shape[1], args.alpha, args.sparse).cuda()
        discriminator1 = Edge_Discriminator(nnodes, nfeats + str_encodings.shape[1], args.alpha, args.sparse).cuda()


        optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=args.lr_gcl, weight_decay=args.w_decay)
        optimizer_discriminator0 = torch.optim.Adam(discriminator0.parameters(), lr=args.lr_disc, weight_decay=args.w_decay)
        optimizer_discriminator1 = torch.optim.Adam(discriminator1.parameters(), lr=args.lr_disc, weight_decay=args.w_decay)

        features = features.cuda()
        str_encodings = str_encodings.cuda()
        edges = edges.cuda()

        best_acc_val = 0
        best_acc_test = 0

        for epoch in range(1, args.epochs + 1):

            for _ in range(args.cl_rounds):
                cl_loss = train_cl(cl_model, discriminator0, discriminator1, optimizer_cl, features, str_encodings, edges)
            rank_loss = train_discriminator(cl_model, discriminator0, optimizer_discriminator0, discriminator1, optimizer_discriminator1, features, str_encodings, edges, args)

            print("[TRAIN] Epoch:{:04d} | CL Loss {:.4f} | RANK loss:{:.4f} ".format(epoch, cl_loss, rank_loss))

            if epoch % args.eval_freq == 0:
                cl_model.eval()
                discriminator0.eval()
                discriminator1.eval()
                adj_1_0, adj_2_0, _, _ = discriminator0(torch.cat((features, str_encodings), 1), edges)
                adj_1_1, adj_2_1, _, _ = discriminator1(torch.cat((features, str_encodings), 1), edges)
                adj_1 = (adj_1_0 + adj_1_1)/2
                adj_2 = (adj_2_0 + adj_2_1)/2

                embedding = cl_model.get_embedding(features, adj_1, adj_2)
                cur_split = 0 if (train_mask.shape[1]==1) else (trial % train_mask.shape[1])
                acc_test, acc_val = eval_test_mode(embedding, labels, train_mask[:, cur_split],
                                                 val_mask[:, cur_split], test_mask[:, cur_split])
                print(
                    '[TEST] Epoch:{:04d} | CL loss:{:.4f} | RANK loss:{:.4f} | VAL ACC:{:.2f} | TEST ACC:{:.2f}'.format(
                        epoch, cl_loss, rank_loss, acc_val, acc_test))

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_acc_test = acc_test

        results.append(best_acc_test)

    print('\n[FINAL RESULT] Dataset:{} | Run:{} | ACC:{:.2f}+-{:.2f}'.format(args.dataset, args.ntrials, np.mean(results),
                                                                           np.std(results)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ESSENTIAL
    parser.add_argument('-dataset', type=str, default='cornell',
                        choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'cornell',
                                 'texas', 'wisconsin', 'computers', 'photo', 'cs', 'physics', 'wikics'])
    parser.add_argument('-ntrials', type=int, default=10)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-eval_freq', type=int, default=20)
    parser.add_argument('-epochs', type=int, default=400)
    parser.add_argument('-lr_gcl', type=float, default=0.001)
    parser.add_argument('-lr_disc', type=float, default=0.001)
    parser.add_argument('-cl_rounds', type=int, default=2)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.5)

    # DISC Module - Hyper-param
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-margin_hom', type=float, default=0.5)
    parser.add_argument('-margin_het', type=float, default=0.5)

    # GRL Module - Hyper-param
    parser.add_argument('-nlayers_enc', type=int, default=2)
    parser.add_argument('-nlayers_proj', type=int, default=1, choices=[1, 2])
    parser.add_argument('-emb_dim', type=int, default=128)
    parser.add_argument('-proj_dim', type=int, default=128)
    parser.add_argument('-cl_batch_size', type=int, default=0)
    parser.add_argument('-k', type=int, default=20)
    parser.add_argument('-maskfeat_rate_1', type=float, default=0.1)
    parser.add_argument('-maskfeat_rate_2', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_1', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_2', type=float, default=0.1)

    args = parser.parse_args()

    print(args)
    main(args)