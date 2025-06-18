from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits


class Diversify(Algorithm):
    def __init__(self, args):
        super(Diversify, self).__init__(args)

        self.featurizer = get_fea(args)

        self.dbottleneck = common_network.feat_bottleneck(
            args.featurizer_out_dim, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)

        self.bottleneck = common_network.feat_bottleneck(
            args.featurizer_out_dim, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(args.bottleneck, args.num_classes)

        self.abottleneck = common_network.feat_bottleneck(
            args.featurizer_out_dim, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(args.bottleneck, args.num_classes * args.latent_domain_num)

        self.dclassifier = None
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)

        self.args = args
        self.dclassifier_initialized = False
        self.domain_labels = None

    def _initialize_dclassifier(self, loader):
        if not self.dclassifier_initialized:
            self.featurizer.eval()
            self.dbottleneck.eval()
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].cuda().float()
                    z1 = self.dbottleneck(self.featurizer(x))
                    z1_dim = z1.shape[1]
                    self.dclassifier = common_network.feat_classifier(z1_dim, self.args.latent_domain_num).cuda()
                    self.dclassifier_initialized = True
                    print(f"[INIT] dclassifier initialized with input dim: {z1_dim}")
                    break
            self.featurizer.train()
            self.dbottleneck.train()

    def update_d(self, minibatch, opt):
        all_x1 = minibatch[0].cuda().float()
        all_c1 = minibatch[1].cuda().long()
        all_d1 = minibatch[2].cuda().long()

        assert all_d1.min() >= 0 and all_d1.max() < self.args.latent_domain_num, \
            f"Domain labels out-of-range [min={all_d1.min()}, max={all_d1.max()}] for {self.args.latent_domain_num} domains"

        z1 = self.dbottleneck(self.featurizer(all_x1))

        assert self.dclassifier_initialized, "dclassifier not initialized!"

        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1)

        cd1 = self.dclassifier(z1)
        ent_loss = Entropylogits(cd1) * self.args.lam + F.cross_entropy(cd1, all_d1)

        loss = ent_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def update_a(self, minibatches, opt):
        all_x = minibatches[0].cuda().float()
        all_c = minibatches[1].cuda().long()
        all_d = minibatches[2].cuda().long()

        assert all_d.min() >= 0 and all_d.max() < self.args.latent_domain_num, \
            f"Domain labels out-of-range in update_a [min={all_d.min()}, max={all_d.max()}]"

        all_y = all_d * self.args.num_classes + all_c

        assert all_y.min() >= 0, "Label contains negative index!"
        assert all_y.max() < self.args.num_classes * self.args.latent_domain_num, "Label exceeds max class index!"

        all_z = self.abottleneck(self.featurizer(all_x))
        all_preds = self.aclassifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)

        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        return {'class': classifier_loss.item()}

    def set_dlabel(self, loader):
        """Placeholder for domain label setting"""
        pass

    def update(self, minibatch, opt):
        """Domain-invariant feature learning step (corrected version with GRL + dis_loss)"""
        all_x = minibatch[0].cuda().float()
        all_y = minibatch[1].cuda().long()
        all_d = minibatch[2].cuda().long()

        # === Feature extraction ===
        features = self.featurizer(all_x)
        features_b = self.bottleneck(features)
        logits = self.classifier(features_b)
        class_loss = F.cross_entropy(logits, all_y)

        # === Domain discriminator via GRL ===
        features_grl = Adver_network.ReverseLayerF.apply(features_b, self.args.alpha)
        d_logits = self.discriminator(features_grl)
        dis_loss = F.cross_entropy(d_logits, all_d)

        # === Total loss ===
        total_loss = class_loss + self.args.lam * dis_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        return {
            'class': class_loss.item(),
            'dis': dis_loss.item(),
            'total': total_loss.item()
        }

    def eval(self):
        super().eval()
        self.featurizer.eval()
        self.dbottleneck.eval()
        self.ddiscriminator.eval()
        self.bottleneck.eval()
        self.classifier.eval()
        self.abottleneck.eval()
        self.aclassifier.eval()
        if self.dclassifier:
            self.dclassifier.eval()

    def train(self, mode=True):
        super().train(mode)
        self.featurizer.train(mode)
        self.dbottleneck.train(mode)
        self.ddiscriminator.train(mode)
        self.bottleneck.train(mode)
        self.classifier.train(mode)
        self.abottleneck.train(mode)
        self.aclassifier.train(mode)
        if self.dclassifier:
            self.dclassifier.train(mode)

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))

    def predict1(self, x):
        return self.ddiscriminator(self.dbottleneck(self.featurizer(x)))
