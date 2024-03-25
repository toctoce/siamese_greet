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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else :
    device = torch.device("cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def train_cl(cl_model0, cl_model1, discriminator0, discriminator1, optimizer_cl0, optimizer_cl1, features, str_encodings, edges):

    cl_model0.train()
    cl_model1.train()
    discriminator0.eval()
    discriminator1.eval()

    adj_10, adj_20, weights_lp0, _ = discriminator0(features, edges)
    adj_11, adj_21, weights_lp1, _ = discriminator1(str_encodings, edges)

    features_10, adj_10, features_20, adj_20 = augmentation(features, adj_10, features, adj_20, args, cl_model0.training)
    features_11, adj_11, features_21, adj_21 = augmentation(features, adj_11, features, adj_21, args, cl_model1.training)

    cl_loss0 = cl_model0(features_10, adj_10, features_20, adj_20)
    cl_loss1 = cl_model1(features_11, adj_11, features_21, adj_21)

    cl_loss = cl_loss0 + cl_loss1

    optimizer_cl0.zero_grad()
    optimizer_cl1.zero_grad()
    cl_loss.backward()
    optimizer_cl0.step()
    optimizer_cl1.step()

    return cl_loss0.item(), cl_loss1.item(), cl_loss.item()


def train_discriminator(cl_model0, cl_model1, discriminator0, discriminator1, emb_layer, optimizer_disc, features, str_encodings, edges, args, flag):

    cl_model0.eval()
    cl_model1.eval()
    discriminator0.train()
    discriminator1.train()
    emb_layer.train()

    adj_10, adj_20, weights_lp0, weights_hp0 = discriminator0(features, edges)
    adj_11, adj_21, weights_lp1, weights_hp1 = discriminator1(str_encodings, edges)

    if flag:
        weights_lp = weights_lp1
        weights_hp = weights_hp1
        discriminator0.eval()
        discriminator1.train()

    else:
        weights_lp = weights_lp0
        weights_hp = weights_hp0
        discriminator0.train()
        discriminator1.eval()


    rand_np = generate_random_node_pairs(features.shape[0], edges.shape[1])
    psu_label = torch.ones(edges.shape[1]).to(device)

    embedding0 = cl_model0.get_embedding(features, adj_10, adj_20)
    embedding1 = cl_model1.get_embedding(features, adj_11, adj_21)
    embedding = torch.cat((embedding0, embedding1), dim=1)
    
    embedding = emb_layer(embedding)

    edge_emb_sim = F.cosine_similarity(embedding[edges[0]], embedding[edges[1]])

    rnp_emb_sim_lp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_lp = F.margin_ranking_loss(edge_emb_sim, rnp_emb_sim_lp, psu_label, margin=args.margin_hom, reduction='none')
    loss_lp *= torch.relu(weights_lp - 0.5)

    rnp_emb_sim_hp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_hp = F.margin_ranking_loss(rnp_emb_sim_hp, edge_emb_sim, psu_label, margin=args.margin_het, reduction='none')
    loss_hp *= torch.relu(weights_hp - 0.5)

    rank_loss = (loss_lp.mean() + loss_hp.mean()) / 2

    optimizer_disc.zero_grad()
    rank_loss.backward()

    optimizer_disc.step()

    return rank_loss.item()


def main(args):

    setup_seed(0)
    features, edges, str_encodings, train_mask, val_mask, test_mask, labels, nnodes, nfeats = load_data(args.dataset)
    results = []
    results_disc0 = []
    results_disc1 = []
    for trial in range(args.ntrials):

        setup_seed(trial)

        cl_model0 = GCL(nlayers=args.nlayers_enc, nlayers_proj=args.nlayers_proj, in_dim=nfeats, emb_dim=args.emb_dim,
                    proj_dim=args.proj_dim, dropout=args.dropout, sparse=args.sparse, batch_size=args.cl_batch_size).to(device)
        cl_model1 = GCL(nlayers=args.nlayers_enc, nlayers_proj=args.nlayers_proj, in_dim=nfeats, emb_dim=args.emb_dim,
                    proj_dim=args.proj_dim, dropout=args.dropout, sparse=args.sparse, batch_size=args.cl_batch_size).to(device)        
        cl_model0.set_mask_knn(features.cpu(), k=args.k, dataset=args.dataset)
        cl_model1.set_mask_knn(features.cpu(), k=args.k, dataset=args.dataset)

        discriminator0 = Edge_Discriminator(nnodes, nfeats, args.alpha, args.sparse).to(device)
        discriminator1 = Edge_Discriminator(nnodes, str_encodings.shape[1], args.alpha, args.sparse).to(device)
        

        optimizer_cl0 = torch.optim.Adam(cl_model0.parameters(), lr=args.lr_gcl, weight_decay=args.w_decay)
        optimizer_cl1= torch.optim.Adam(cl_model1.parameters(), lr=args.lr_gcl, weight_decay=args.w_decay)

        optimizer_discriminator0 = torch.optim.Adam(discriminator0.parameters(), lr=args.lr_disc, weight_decay=args.w_decay)
        optimizer_discriminator1 = torch.optim.Adam(discriminator1.parameters(), lr=args.lr_disc, weight_decay=args.w_decay)

        emb_layer = nn.Linear(args.emb_dim * 4, args.emb_dim).to(device)

        features = features.to(device)
        str_encodings = str_encodings.to(device)
        edges = edges.to(device)

        best_acc_val = 0
        best_acc_test = 0
        best_acc_disc0 = 0
        best_acc_disc1 = 0

        for epoch in range(1, args.epochs + 1):

            for _ in range(args.cl_rounds):
                cl_loss0, cl_loss1, cl_loss = train_cl(cl_model0, cl_model1, discriminator0, discriminator1, optimizer_cl0, optimizer_cl1, features, str_encodings, edges)
            rank_loss0 = train_discriminator(cl_model0, cl_model1, discriminator0, discriminator1, emb_layer, optimizer_discriminator0, features, str_encodings, edges, args, 0)
            rank_loss1 = train_discriminator(cl_model0, cl_model1, discriminator0, discriminator1, emb_layer, optimizer_discriminator1, features, str_encodings, edges, args, 1)

            print("[TRAIN] Epoch:{:04d} | CL Loss0 {:.4f} | CL Loss1 {:.4f} | RANK loss0:{:.4f} | RANK loss1:{:.4f}".format(epoch, cl_loss0, cl_loss1, rank_loss0, rank_loss1))


            if epoch % args.eval_freq == 0:
                cl_model0.eval()
                cl_model1.eval()
                discriminator0.eval()
                discriminator1.eval()
                emb_layer.eval()

                adj_10, adj_20, weights_lp0, weights_hp0 = discriminator0(features, edges)
                adj_11, adj_21, weights_lp1, weights_hp1 = discriminator1(str_encodings, edges)
                acc_disc0 = accuracy_discriminator(edges, labels, weights_lp0)
                acc_disc1 = accuracy_discriminator(edges, labels, weights_lp1)

                embedding0 = cl_model0.get_embedding(features, adj_10, adj_20)
                embedding1 = cl_model1.get_embedding(features, adj_11, adj_21)
                embedding = torch.cat((embedding0, embedding1), dim=1)

                embedding = emb_layer(embedding)
                
                cur_split = 0 if (train_mask.shape[1]==1) else (trial % train_mask.shape[1])
                acc_test, acc_val = eval_test_mode(embedding, labels, train_mask[:, cur_split],
                                                 val_mask[:, cur_split], test_mask[:, cur_split])
                print(
                    '[TEST] Epoch:{:04d} | CL loss:{:.4f} | RANK loss0:{:.4f} | RANK loss1:{:.4f} | VAL ACC:{:.2f} | TEST ACC:{:.2f} | DISC0 ACC:{:.2f} | DISC1 ACC:{:.2f}'.format(
                        epoch, cl_loss, rank_loss0, rank_loss1, acc_val, acc_test, acc_disc0, acc_disc1))

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_acc_test = acc_test
                    best_acc_disc0 = acc_disc0
                    best_acc_disc1 = acc_disc1


        results.append(best_acc_test)
        results_disc0.append(best_acc_disc0)
        results_disc1.append(best_acc_disc1)

    print('\n[FINAL RESULT] Dataset:{} | Run:{} | ACC:{:.2f}+-{:.2f} | DISC0 ACC:{:.2f}+-{:.2f} | DISC1 ACC:{:.2f}+-{:.2f}'.format(args.dataset, args.ntrials, np.mean(results),
                                                                           np.std(results), np.mean(results_disc0), np.std(results_disc0), np.mean(results_disc1), np.std(results_disc1)))

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