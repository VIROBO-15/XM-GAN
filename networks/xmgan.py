import random

import numpy as np
from torch import autograd

from networks.blocks import *
from networks.loss import *
from utils import batched_index_select, batched_scatter
from featurefusionnetwork import *
from networks.transformer import TransformerUnit
import torch
from torchvision.utils import save_image
#from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
#import math

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

class xmgan(nn.Module):
    def __init__(self, config):
        super(xmgan, self).__init__()

        self.gen = Generator(config['gen'])
        self.dis = Discriminator(config['dis'])

        self.w_adv_g = config['w_adv_g']
        self.w_adv_d = config['w_adv_d']
        self.w_recon = config['w_recon']
        self.w_cls = config['w_cls']
        self.w_gp = config['w_gp']
        self.n_sample = config['n_sample_train']
        self.vgg_loss = VGGLoss()

    def forward(self, xs, y, it,  mode):

        alphas = torch.rand((1)).cuda().repeat(xs.shape[0],1) #torch.rand((xs.shape[0], 1)).cuda()#
    

        if mode == 'gen_update':

            
            fake_x, base_index, r1_imgs, r2_imgs = self.gen(xs, alphas, base_index = None)

            # loss_recon = local_recon_criterion(xs, fake_x, similarity, indices_feat, indices_ref, base_index, s=8)
            # for i in range(8):
            #     a = int(y[i])
            #     img = (fake_x[i] + 1)/2
            #     save_image(img, f'/proj/cvl/users/x_fahkh/mn/medical_img/generated_img/class_{a}_iter_{it}.jpg')
            # print(y)

            feat_real, _, _ = self.dis(xs)
            feat_fake, logit_adv_fake, logit_c_fake = self.dis(fake_x)
            loss_adv_gen = torch.mean(-logit_adv_fake)
            loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())

            #loss_recon = loss_recon * self.w_recon
            loss_adv_gen = loss_adv_gen * self.w_adv_g
            loss_cls_gen = loss_cls_gen * self.w_cls

            loss_vgg = (alphas*self.vgg_loss(fake_x, r1_imgs).unsqueeze(1)) + \
                       ((1-alphas)*self.vgg_loss(fake_x, r2_imgs).unsqueeze(1))

            loss_vgg = 50*loss_vgg.mean()

            #loss_total = loss_adv_gen + loss_cls_gen
            loss_total = loss_adv_gen + loss_cls_gen + loss_vgg
            
            loss_total.backward()

            return {'loss_total': loss_total,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen,
                    'loss_vgg':loss_vgg}

        elif mode == 'dis_update':
            xs.requires_grad_()

            _, logit_adv_real, logit_c_real = self.dis(xs)
            loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
            loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d
            loss_adv_dis_real.backward(retain_graph=True)

            y_extend = y.repeat(1, self.n_sample).view(-1)
            index = torch.LongTensor(range(y_extend.size(0))).cuda()
            logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)
            loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)

            loss_reg_dis = loss_reg_dis * self.w_gp
            loss_reg_dis.backward(retain_graph=True)

            loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
            loss_cls_dis = loss_cls_dis * self.w_cls
            loss_cls_dis.backward()

            with torch.no_grad():
                fake_x = self.gen(xs, alphas)[0]

            _, logit_adv_fake, _ = self.dis(fake_x.detach())
            loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
            loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d
            loss_adv_dis_fake.backward()

            loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
            return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis,
                    'loss_reg': loss_reg_dis}

        else:
            assert 0, 'Not support operation'

    def generate(self, xs, alphas = None, base_index = None, noise = None):

        if alphas == None:
            alphas =  torch.rand((1)).cuda().repeat(xs.shape[0],1)#torch.rand((xs.shape[0], 1)).cuda()#

        fake_x = self.gen(xs, alphas, base_index, noise)[0]    

        return fake_x

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()
        reg /= batch_size
        return reg


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.soft_label = False
        nf = config['nf']
        n_class = config['num_classes']
        n_res_blks = config['n_res_blks']

        cnn_f = [Conv2dBlock(3, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        for i in range(n_res_blks):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        cnn_adv = [nn.AdaptiveAvgPool2d(1),
                   Conv2dBlock(nf_out, 1, 1, 1,
                               norm='none',
                               activation='none',
                               activation_first=False)]
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(nf_out, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_adv = nn.Sequential(*cnn_adv)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x):
        if len(x.size()) == 5:
            B, K, C, H, W = x.size()
            x = x.view(B * K, C, H, W)
        else:
            B, C, H, W = x.size()
            K = 1
        feat = self.cnn_f(x)
        logit_adv = self.cnn_adv(feat).view(B * K, -1)
        logit_c = self.cnn_c(feat).view(B * K, -1)
        return feat, logit_adv, logit_c


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        d_m = 256
        d_inner = 128
        n_layer = 1
        nhead = 8
        self.meta_conv_block = Conv2dBlock(d_m, d_m, 1, 1, 0,
                                norm='bn',
                                activation='lrelu',
                                pad_type='reflect')

        self.global_avg_pool = GlobalAvgPool2d()
        
        self.mlp = nn.Linear(128,  d_m)#MLPconv(64 * d_m)#'
        # self.embalpha_ = nn.Embedding(100+1,  64 * d_m)
        # self.embalpha = nn.Embedding(100+1,  64 * d_m)
        self.embalpha_ = nn.Linear(1, d_m)
        self.embalpha = nn.Linear(1, d_m)

        self.encoder = Encoder(d_m = d_m)
        self.decoder = Decoder(d_m = d_m)


        self.cbi_br1  = TransformerUnit(n_layers=n_layer, n_head=nhead, d_k=d_m, d_v=d_m,
            d_model=d_m, d_inner=d_inner, )


    def forward(self, xs, alphas = None, base_index = None, noise = None):

    
        b, k, C, H, W = xs.size()

        if alphas is None:

            alphas = torch.rand((1)).cuda().repeat(xs.shape[0],1)

        if noise is None:

            noise = torch.FloatTensor(np.random.normal(0, 1, (b, 128))).cuda()

        
        # w1 = w + w*self.embalpha.weight[(alphas*100).long().squeeze(1)]  + self.embalpha_.weight[(alphas*100).long().squeeze(1)]
        # w2 = w + w*self.embalpha.weight[((1-alphas)*100).long().squeeze(1)]  + self.embalpha_.weight[((1-alphas)*100).long().squeeze(1)]

        # if torch.rand(1).item()>0.5:

        #     w1 = w + alphas*self.f(w)
        #     w2 = w + (1-alphas)*self.f(w)

        # else:

        #     w1 = w + self.f(w)
        #     w2 = w + self.f(w)


        xs = xs.view(-1, C, H, W)
        querys = self.encoder(xs)
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)

        if base_index == None:

            base_index = random.choice(range(k))

        base_feat = querys[:, base_index, :, :, :]

        rf_index = [i for i in range(k) if i!=base_index]

        r1_feat = querys[:, rf_index[0], :, :, :]
        r2_feat = querys[:, rf_index[1], :, :, :]


        
        w_noise = self.mlp(noise)

        w1 = w_noise + w_noise * self.embalpha_(alphas) * self.global_avg_pool(self.meta_conv_block(r1_feat)) 
        w2 = w_noise +  w_noise * self.embalpha_(1-alphas) * self.global_avg_pool(self.meta_conv_block(r2_feat)) 

        w1 = w1.view(b, -1, 256)
        w2 = w2.view(b, -1, 256)
        
        base_feat = base_feat.view(b, c, -1).permute(0,2,1)
        r1_feat = r1_feat.view(b, c, -1).permute(0,2,1)
        r2_feat = r2_feat.view(b, c, -1).permute(0,2,1)

        base_r1_feat = self.cbi_br1(query = base_feat, key = r1_feat, value = r1_feat, z = w1)        
        base_r2_feat = self.cbi_br1(query = base_feat, key = r2_feat, value = r2_feat, z = w2)

        fuse_feature = alphas.unsqueeze(-1)*base_r1_feat + (1-alphas.unsqueeze(-1))*base_r2_feat

        fuse_feature = fuse_feature.permute(0, 2, 1)
        fuse_feature = fuse_feature.view(b, -1, h, w)

        fake_x = self.decoder(fuse_feature)

        imgs = xs.view(b, k, C, H, W)
        r1_imgs = imgs[:, rf_index[0], :, :, :]
        r2_imgs = imgs[:, rf_index[1], :, :, :]

        return fake_x, base_index, r1_imgs, r2_imgs


class Encoder(nn.Module):
    def __init__(self, d_m = 128):
        super(Encoder, self).__init__()

        model = [Conv2dBlock(3, 32, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(32, 64, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(64, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(128, d_m, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, d_m = 128):
        super(Decoder, self).__init__()

        model = [nn.Upsample(scale_factor=2),
                 Conv2dBlock(d_m, d_m, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 nn.Upsample(scale_factor=2),
                 Conv2dBlock(d_m, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 nn.Upsample(scale_factor=2),
                 Conv2dBlock(128, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 nn.Upsample(scale_factor=2),
                 Conv2dBlock(64, 32, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(32, 3, 5, 1, 2,
                             norm='none',
                             activation='tanh',
                             pad_type='reflect')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class LocalFusionModule(nn.Module):
    def __init__(self, inplanes, rate):
        super(LocalFusionModule, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate

    def forward(self, feat, refs, index, similarity):
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)

        # take ref:(32, 2, 128, 8, 8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)

        # local selection
        rate = self.rate
        num = int(rate * h * w)
        feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],
                                 dim=0).cuda()  # B*num

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=feat_indices)  # (32*128*12)

        # local matching
        w_feat_select = batched_index_select(w_feat, dim=1, index=feat_indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)
            select = batched_index_select(ref, dim=2, index=indice)  # (32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)

        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        feat = batched_scatter(feat, dim=2, index=feat_indices, src=feat_fused)
        feat = feat.view(b, c, h, w)  # (32*128*8*8)

        return feat, feat_indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)


if __name__ == '__main__':
    config = {}
    model = Generator(config).cuda()
    x = torch.randn(32, 3, 3, 128, 128).cuda()
    y, sim = model(x)
    print(y.size())
