from typing import Optional
from itertools import chain
from functools import partial

import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from utils import build_args

args = build_args()
device = args.device if args.device >= 0 else "cpu"
class Encoder(nn.Module):
    def __init__(self, dropout= 0.5,num_class = 1, nodes=64):
        super().__init__()
        self.k = 8
        self.attention_path = nn.Linear(nodes*nodes, 1)
        self.self_attention_norm = nn.LayerNorm(48)
        self.self_attention_dropout = nn.Dropout(dropout)
        self.n2g_global = nn.Sequential(
            nn.Conv2d(48, nodes, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )
        '''all subnetwork'''
        self.e2e_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_2 = nn.LeakyReLU(0.33)
        self.e2e_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_4 = nn.LeakyReLU(0.33)
        self.e2e_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_6 = nn.LeakyReLU(0.33)
        self.e2e_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_8 = nn.LeakyReLU(0.33)

        self.e2n_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1,
                                    groups=1, bias=True, K=self.k)
        self.e2n_2 = nn.LeakyReLU(0.33)

        '''single subnetwork DMN-DMN'''
        self.e2e_DMN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DMN_2 = nn.LeakyReLU(0.33)
        self.e2e_DMN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_4 = nn.LeakyReLU(0.33)
        self.e2e_DMN_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_6 = nn.LeakyReLU(0.33)
        self.e2e_DMN_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_8 = nn.LeakyReLU(0.33)


        self.e2n_DMN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DMN_2 = nn.LeakyReLU(0.33)

        '''single subnetwork CEN-CEN'''
        self.e2e_CEN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CEN_2 = nn.LeakyReLU(0.33)
        self.e2e_CEN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CEN_4 = nn.LeakyReLU(0.33)
        self.e2e_CEN_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_CEN_6 = nn.LeakyReLU(0.33)
        self.e2e_CEN_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_CEN_8 = nn.LeakyReLU(0.33)

        self.e2n_CEN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CEN_2 = nn.LeakyReLU(0.33)

        '''single subnetwork SN-SN'''
        self.e2e_SN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SN_2 = nn.LeakyReLU(0.33)
        self.e2e_SN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SN_4 = nn.LeakyReLU(0.33)
        self.e2e_SN_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_SN_6 = nn.LeakyReLU(0.33)
        self.e2e_SN_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_SN_8= nn.LeakyReLU(0.33)

        self.e2n_SN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                       dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SN_2 = nn.LeakyReLU(0.33)

        '''DMN-CEN or DMN-CEN-DMN'''
        self.e2e_DMN_CEN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DMN_CEN_2 = nn.LeakyReLU(0.33)
        self.e2e_DMN_CEN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_CEN_4 = nn.LeakyReLU(0.33)
        self.e2e_DMN_CEN_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_CEN_6 = nn.LeakyReLU(0.33)
        self.e2e_DMN_CEN_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_CEN_8 = nn.LeakyReLU(0.33)

        self.e2n_DMN_CEN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                            dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DMN_CEN_2 = nn.LeakyReLU(0.33)

        '''DMN-SN or DMN-SN-DMN'''
        self.e2e_DMN_SN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DMN_SN_2 = nn.LeakyReLU(0.33)
        self.e2e_DMN_SN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_SN_4 = nn.LeakyReLU(0.33)
        self.e2e_DMN_SN_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_SN_6 = nn.LeakyReLU(0.33)
        self.e2e_DMN_SN_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_SN_8 = nn.LeakyReLU(0.33)

        self.e2n_DMN_SN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                           dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DMN_SN_2 = nn.LeakyReLU(0.33)

        '''CEN-SN or CEN-SN-CEN'''
        self.e2e_CEN_SN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CEN_SN_2 = nn.LeakyReLU(0.33)
        self.e2e_CEN_SN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CEN_SN_4 = nn.LeakyReLU(0.33)
        self.e2e_CEN_SN_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_CEN_SN_6 = nn.LeakyReLU(0.33)
        self.e2e_CEN_SN_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_CEN_SN_8 = nn.LeakyReLU(0.33)

        self.e2n_CEN_SN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                           dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CEN_SN_2 = nn.LeakyReLU(0.33)

        '''DMN-CEN-SN'''
        self.e2e_DCS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DCS_2 = nn.LeakyReLU(0.33)
        self.e2e_DCS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DCS_4 = nn.LeakyReLU(0.33)
        self.e2e_DCS_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DCS_6 = nn.LeakyReLU(0.33)
        self.e2e_DCS_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DCS_8 = nn.LeakyReLU(0.33)

        self.e2n_DCS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DCS_2 = nn.LeakyReLU(0.33)

        '''DMN-SN-CEN'''
        self.e2e_DSC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DSC_2 = nn.LeakyReLU(0.33)
        self.e2e_DSC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DSC_4 = nn.LeakyReLU(0.33)
        self.e2e_DSC_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DSC_6 = nn.LeakyReLU(0.33)
        self.e2e_DSC_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DSC_8 = nn.LeakyReLU(0.33)

        self.e2n_DSC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DSC_2 = nn.LeakyReLU(0.33)

        '''CEN-DMN-SN'''
        self.e2e_CDS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CDS_2 = nn.LeakyReLU(0.33)
        self.e2e_CDS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CDS_4 = nn.LeakyReLU(0.33)
        self.e2e_CDS_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_CDS_6 = nn.LeakyReLU(0.33)
        self.e2e_CDS_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_CDS_8 = nn.LeakyReLU(0.33)

        self.e2n_CDS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CDS_2 = nn.LeakyReLU(0.33)

        '''DMN-DMN-CEN'''
        self.e2e_DDC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDC_2 = nn.LeakyReLU(0.33)
        self.e2e_DDC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDC_4 = nn.LeakyReLU(0.33)
        self.e2e_DDC_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDC_6 = nn.LeakyReLU(0.33)
        self.e2e_DDC_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDC_8 = nn.LeakyReLU(0.33)

        self.e2n_DDC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDC_2 = nn.LeakyReLU(0.33)

        '''DMN-DMN-SN'''
        self.e2e_DDS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDS_2 = nn.LeakyReLU(0.33)
        self.e2e_DDS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDS_4 = nn.LeakyReLU(0.33)
        self.e2e_DDS_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDS_6 = nn.LeakyReLU(0.33)
        self.e2e_DDS_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDS_8 = nn.LeakyReLU(0.33)

        self.e2n_DDS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDS_2 = nn.LeakyReLU(0.33)

        '''CEN-CEN-DMN'''
        self.e2e_CCD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCD_2 = nn.LeakyReLU(0.33)
        self.e2e_CCD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCD_4 = nn.LeakyReLU(0.33)
        self.e2e_CCD_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCD_6 = nn.LeakyReLU(0.33)
        self.e2e_CCD_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCD_8 = nn.LeakyReLU(0.33)

        self.e2n_CCD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCD_2 = nn.LeakyReLU(0.33)

        '''CEN-CEN-SN'''
        self.e2e_CCS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCS_2 = nn.LeakyReLU(0.33)
        self.e2e_CCS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCS_4 = nn.LeakyReLU(0.33)
        self.e2e_CCS_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCS_6 = nn.LeakyReLU(0.33)
        self.e2e_CCS_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCS_8 = nn.LeakyReLU(0.33)

        self.e2n_CCS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCS_2 = nn.LeakyReLU(0.33)

        '''SN-SN-DMN'''
        self.e2e_SSD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SSD_2 = nn.LeakyReLU(0.33)
        self.e2e_SSD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSD_4 = nn.LeakyReLU(0.33)
        self.e2e_SSD_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSD_6 = nn.LeakyReLU(0.33)
        self.e2e_SSD_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSD_8 = nn.LeakyReLU(0.33)

        self.e2n_SSD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SSD_2 = nn.LeakyReLU(0.33)

        '''SN-SN-CEN'''
        self.e2e_SSC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SSC_2 = nn.LeakyReLU(0.33)
        self.e2e_SSC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSC_4 = nn.LeakyReLU(0.33)
        self.e2e_SSC_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSC_6 = nn.LeakyReLU(0.33)
        self.e2e_SSC_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSC_8 = nn.LeakyReLU(0.33)

        self.e2n_SSC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                        dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SSC_2 = nn.LeakyReLU(0.33)

        '''DMN-DMN-CEN-CEN'''
        self.e2e_DDCC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDCC_2 = nn.LeakyReLU(0.33)
        self.e2e_DDCC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDCC_4 = nn.LeakyReLU(0.33)
        self.e2e_DDCC_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDCC_6 = nn.LeakyReLU(0.33)
        self.e2e_DDCC_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDCC_8 = nn.LeakyReLU(0.33)

        self.e2n_DDCC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDCC_2 = nn.LeakyReLU(0.33)

        '''DMN-DMN-SN-SN'''
        self.e2e_DDSS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDSS_2 = nn.LeakyReLU(0.33)
        self.e2e_DDSS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDSS_4 = nn.LeakyReLU(0.33)
        self.e2e_DDSS_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDSS_6 = nn.LeakyReLU(0.33)
        self.e2e_DDSS_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDSS_8 = nn.LeakyReLU(0.33)

        self.e2n_DDSS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDSS_2 = nn.LeakyReLU(0.33)

        '''CEN-CEN-SN-SN'''
        self.e2e_CCSS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCSS_2 = nn.LeakyReLU(0.33)
        self.e2e_CCSS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCSS_4 = nn.LeakyReLU(0.33)
        self.e2e_CCSS_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCSS_6 = nn.LeakyReLU(0.33)
        self.e2e_CCSS_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCSS_8 = nn.LeakyReLU(0.33)

        self.e2n_CCSS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCSS_2 = nn.LeakyReLU(0.33)

        '''DMN-DMN-CEN-SN'''
        self.e2e_DDCS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDCS_2 = nn.LeakyReLU(0.33)
        self.e2e_DDCS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDCS_4 = nn.LeakyReLU(0.33)
        self.e2e_DDCS_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDCS_6 = nn.LeakyReLU(0.33)
        self.e2e_DDCS_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDCS_8 = nn.LeakyReLU(0.33)

        self.e2n_DDCS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDCS_2 = nn.LeakyReLU(0.33)

        '''DMN-DMN-SN-CEN'''
        self.e2e_DDSC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDSC_2 = nn.LeakyReLU(0.33)
        self.e2e_DDSC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDSC_4 = nn.LeakyReLU(0.33)
        self.e2e_DDSC_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDSC_6 = nn.LeakyReLU(0.33)
        self.e2e_DDSC_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDSC_8 = nn.LeakyReLU(0.33)

        self.e2n_DDSC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDSC_2 = nn.LeakyReLU(0.33)

        '''CEN-CEN-DMN-SN'''
        self.e2e_CCDS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCDS_2 = nn.LeakyReLU(0.33)
        self.e2e_CCDS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCDS_4 = nn.LeakyReLU(0.33)
        self.e2e_CCDS_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCDS_6 = nn.LeakyReLU(0.33)
        self.e2e_CCDS_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCDS_8 = nn.LeakyReLU(0.33)

        self.e2n_CCDS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCDS_2 = nn.LeakyReLU(0.33)

        '''CEN-CEN-SN-DMN'''
        self.e2e_CCSD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCSD_2 = nn.LeakyReLU(0.33)
        self.e2e_CCSD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCSD_4 = nn.LeakyReLU(0.33)
        self.e2e_CCSD_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCSD_6 = nn.LeakyReLU(0.33)
        self.e2e_CCSD_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCSD_8 = nn.LeakyReLU(0.33)

        self.e2n_CCSD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCSD_2 = nn.LeakyReLU(0.33)

        '''SN-SN-DMN-CEN'''
        self.e2e_SSDC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SSDC_2 = nn.LeakyReLU(0.33)
        self.e2e_SSDC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSDC_4 = nn.LeakyReLU(0.33)
        self.e2e_SSDC_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSDC_6 = nn.LeakyReLU(0.33)
        self.e2e_SSDC_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSDC_8 = nn.LeakyReLU(0.33)

        self.e2n_SSDC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SSDC_2 = nn.LeakyReLU(0.33)

        '''SN-SN-CEN-DMN'''
        self.e2e_SSCD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SSCD_2 = nn.LeakyReLU(0.33)
        self.e2e_SSCD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSCD_4 = nn.LeakyReLU(0.33)
        self.e2e_SSCD_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSCD_6 = nn.LeakyReLU(0.33)
        self.e2e_SSCD_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSCD_8 = nn.LeakyReLU(0.33)

        self.e2n_SSCD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SSCD_2 = nn.LeakyReLU(0.33)

        '''DMN-SN-CEN-DMN'''
        self.e2e_DSCD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DSCD_2 = nn.LeakyReLU(0.33)
        self.e2e_DSCD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DSCD_4 = nn.LeakyReLU(0.33)
        self.e2e_DSCD_5 = E2E(8, 8, (nodes, nodes))
        self.e2e_DSCD_6 = nn.LeakyReLU(0.33)
        self.e2e_DSCD_7 = E2E(8, 8, (nodes, nodes))
        self.e2e_DSCD_8 = nn.LeakyReLU(0.33)


        self.e2n_DSCD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0,
                                         dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DSCD_2 = nn.LeakyReLU(0.33)

        self.weight_attention = nn.Parameter(torch.Tensor(64, 26))
        self.bias_attention = nn.Parameter(torch.Tensor(1))

        self.linear = nn.Sequential(
            nn.Linear(nodes * nodes, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(64, 10),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(10, 2)
        )
        self.reset_parameters()

        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_attention, a=0)
        nn.init.zeros_(self.bias_attention)

    def forward(self, x, x_metapath):
        x = x.float()
        # '''single subnetwork'''
        x_DMN = x_metapath['DMN'].float()
        x_CEN = x_metapath['CEN'].float()
        x_SN = x_metapath['SN'].float()
        
        # '''DMN-CEN or DMN-SN or CEN-SN'''
        x_DMN_CEN = x_metapath['DMN_CEN'].float()
        x_DMN_SN = x_metapath['DMN_SN'].float()
        x_CEN_SN = x_metapath['CEN_SN'].float()

        x_DCS = x_metapath['DMN_CEN_SN'].float()
        x_DSC = x_metapath['DMN_SN_CEN'].float()
        x_CDS = x_metapath['CEN_DMN_SN'].float()
        x_DDC = x_metapath['DMN_DMN_CEN'].float()
        x_DDS = x_metapath['DMN_DMN_SN'].float()
        x_CCD = x_metapath['CEN_CEN_DMN'].float()
        x_CCS = x_metapath['CEN_CEN_SN'].float()
        x_SSD = x_metapath['SN_SN_DMN'].float()
        x_SSC = x_metapath['SN_SN_CEN'].float()

        x_DDCC = x_metapath['DMN_DMN_CEN_CEN'].float()
        x_DDSS = x_metapath['DMN_DMN_SN_SN'].float()
        x_CCSS = x_metapath['CEN_CEN_SN_SN'].float()
        x_DDCS = x_metapath['DMN_DMN_CEN_SN'].float()
        x_DDSC = x_metapath['DMN_DMN_SN_CEN'].float()
        x_CCDS = x_metapath['CEN_CEN_DMN_SN'].float()
        x_CCSD = x_metapath['CEN_CEN_SN_DMN'].float()
        x_SSDC = x_metapath['SN_SN_DMN_CEN'].float()
        x_SSCD = x_metapath['SN_SN_CEN_DMN'].float()
        x_DSCD = x_metapath['DMN_SN_CEN_DMN'].float()
        x_att = torch.cat((x.reshape(x.size(0), 1, x.size(1), x.size(2)),
                           x_DMN.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_CEN.reshape(x_CEN.size(0), 1, x_CEN.size(1), x_CEN.size(2)),
                           x_SN.reshape(x_SN.size(0), 1, x_SN.size(1), x_SN.size(2)),
                           x_DMN_CEN.reshape(x_DMN_CEN.size(0), 1, x_DMN_CEN.size(1), x_DMN_CEN.size(2)),
                           x_DMN_SN.reshape(x_DMN_SN.size(0), 1, x_DMN_SN.size(1), x_DMN_SN.size(2)),
                           x_CEN_SN.reshape(x_CEN_SN.size(0), 1, x_CEN_SN.size(1), x_CEN_SN.size(2)),
                           x_DCS.reshape(x_DCS.size(0), 1, x_DCS.size(1), x_DCS.size(2)),
                           x_DSC.reshape(x_DSC.size(0), 1, x_DSC.size(1), x_DSC.size(2)),
                           x_CDS.reshape(x_CDS.size(0), 1, x_CDS.size(1), x_CDS.size(2)),
                           x_DDC.reshape(x_DDC.size(0), 1, x_DDC.size(1), x_DDC.size(2)),
                           x_DDS.reshape(x_DDS.size(0), 1, x_DDS.size(1), x_DDS.size(2)),
                           x_CCD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_CCS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_SSD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_SSC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_DDCC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_DDSS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_CCSS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_DDCS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_DDSC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_CCDS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_CCSD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_SSDC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_SSCD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)),
                           x_DSCD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2))),
                          axis=1)
    
        x_att = self.attention_path(x_att.reshape(x_att.size(0) * x_att.size(1), -1))
        x_att = x_att.reshape(x.size(0), -1)
        x_att = torch.sigmoid(x_att) 

        '''all subnetworks'''
        x = self.e2e_1(x, x_att[:, 0])
        x = self.e2e_2(x)
        x = self.e2e_3(x, x_att[:, 0])
        x = self.e2e_4(x)

        x = self.e2n_1(x, x_att[:, 0]) 
        x = self.e2n_2(x)

        '''single subnetwork'''
        x_DMN = self.e2e_DMN_1(x_DMN, x_att[:, 1])
        x_DMN = self.e2e_DMN_2(x_DMN)
        x_DMN = self.e2e_DMN_3(x_DMN, x_att[:, 1])
        x_DMN = self.e2e_DMN_4(x_DMN)
        x_DMN = self.e2n_DMN_1(x_DMN, x_att[:, 1])  
        x_DMN = self.e2n_DMN_2(x_DMN)

        x_CEN = self.e2e_CEN_1(x_CEN, x_att[:, 2])
        x_CEN = self.e2e_CEN_2(x_CEN)
        x_CEN = self.e2e_CEN_3(x_CEN, x_att[:, 2])
        x_CEN = self.e2e_CEN_4(x_CEN)
        x_CEN = self.e2n_CEN_1(x_CEN, x_att[:, 2])  
        x_CEN = self.e2n_CEN_2(x_CEN)

        x_SN = self.e2e_SN_1(x_SN, x_att[:, 3])
        x_SN = self.e2e_SN_2(x_SN)
        x_SN = self.e2e_SN_3(x_SN, x_att[:, 3])
        x_SN = self.e2e_SN_4(x_SN)
        x_SN = self.e2n_SN_1(x_SN, x_att[:, 3])  
        x_SN = self.e2n_SN_2(x_SN)

        '''length = 1 subnetwork'''
        x_DMN_CEN = self.e2e_DMN_CEN_1(x_DMN_CEN, x_att[:, 4])
        x_DMN_CEN = self.e2e_DMN_CEN_2(x_DMN_CEN)
        x_DMN_CEN = self.e2e_DMN_CEN_3(x_DMN_CEN, x_att[:, 4])
        x_DMN_CEN = self.e2e_DMN_CEN_4(x_DMN_CEN)
        x_DMN_CEN = self.e2n_DMN_CEN_1(x_DMN_CEN, x_att[:, 4])  
        x_DMN_CEN = self.e2n_DMN_CEN_2(x_DMN_CEN)

        x_DMN_SN = self.e2e_DMN_SN_1(x_DMN_SN, x_att[:, 5])
        x_DMN_SN = self.e2e_DMN_SN_2(x_DMN_SN)
        x_DMN_SN = self.e2e_DMN_SN_3(x_DMN_SN, x_att[:, 5])
        x_DMN_SN = self.e2e_DMN_SN_4(x_DMN_SN)
        x_DMN_SN = self.e2n_DMN_SN_1(x_DMN_SN, x_att[:, 5]) 
        x_DMN_SN = self.e2n_DMN_SN_2(x_DMN_SN)

        x_CEN_SN = self.e2e_CEN_SN_1(x_CEN_SN, x_att[:, 6])
        x_CEN_SN = self.e2e_CEN_SN_2(x_CEN_SN)
        x_CEN_SN = self.e2e_CEN_SN_3(x_CEN_SN, x_att[:, 6])
        x_CEN_SN = self.e2e_CEN_SN_4(x_CEN_SN)

        x_CEN_SN = self.e2n_CEN_SN_1(x_CEN_SN, x_att[:, 6])  
        x_CEN_SN = self.e2n_CEN_SN_2(x_CEN_SN)

        '''length = 2 subnetork'''
        x_DCS = self.e2e_DCS_1(x_DCS, x_att[:, 7])
        x_DCS = self.e2e_DCS_2(x_DCS)
        x_DCS = self.e2e_DCS_3(x_DCS, x_att[:, 7])
        x_DCS = self.e2e_DCS_4(x_DCS)
        x_DCS = self.e2n_DCS_1(x_DCS, x_att[:, 7])  
        x_DCS = self.e2n_DCS_2(x_DCS)

        x_DSC = self.e2e_DSC_1(x_DSC, x_att[:, 8])
        x_DSC = self.e2e_DSC_2(x_DSC)
        x_DSC = self.e2e_DSC_3(x_DSC, x_att[:, 8])
        x_DSC = self.e2e_DSC_4(x_DSC)
        x_DSC = self.e2n_DSC_1(x_DSC, x_att[:, 8])  
        x_DSC = self.e2n_DSC_2(x_DSC)

        x_CDS = self.e2e_CDS_1(x_CDS, x_att[:, 9])
        x_CDS = self.e2e_CDS_2(x_CDS)
        x_CDS = self.e2e_CDS_3(x_CDS, x_att[:, 9])
        x_CDS = self.e2e_CDS_4(x_CDS)
        x_CDS = self.e2n_CDS_1(x_CDS, x_att[:, 9])  
        x_CDS = self.e2n_CDS_2(x_CDS)

        x_DDC = self.e2e_DDC_1(x_DDC, x_att[:, 10])
        x_DDC = self.e2e_DDC_2(x_DDC)
        x_DDC = self.e2e_DDC_3(x_DDC, x_att[:, 10])
        x_DDC = self.e2e_DDC_4(x_DDC)
        x_DDC = self.e2n_DDC_1(x_DDC, x_att[:, 10])  
        x_DDC = self.e2n_DDC_2(x_DDC)

        x_DDS = self.e2e_DDS_1(x_DDS, x_att[:, 11])
        x_DDS = self.e2e_DDS_2(x_DDS)
        x_DDS = self.e2e_DDS_3(x_DDS, x_att[:, 11])
        x_DDS = self.e2e_DDS_4(x_DDS)
        x_DDS = self.e2n_DDS_1(x_DDS, x_att[:, 11])  
        x_DDS = self.e2n_DDS_2(x_DDS)

        x_CCD = self.e2e_CCD_1(x_CCD, x_att[:, 12])
        x_CCD = self.e2e_CCD_2(x_CCD)
        x_CCD = self.e2e_CCD_3(x_CCD, x_att[:, 12])
        x_CCD = self.e2e_CCD_4(x_CCD)
        x_CCD = self.e2n_CCD_1(x_CCD, x_att[:, 12])  
        x_CCD = self.e2n_CCD_2(x_CCD)

        x_CCS = self.e2e_CCS_1(x_CCS, x_att[:, 13])
        x_CCS = self.e2e_CCS_2(x_CCS)
        x_CCS = self.e2e_CCS_3(x_CCS, x_att[:, 13])
        x_CCS = self.e2e_CCS_4(x_CCS)
        x_CCS = self.e2n_CCS_1(x_CCS, x_att[:, 13]) 
        x_CCS = self.e2n_CCS_2(x_CCS)

        x_SSD = self.e2e_SSD_1(x_SSD, x_att[:, 14])
        x_SSD = self.e2e_SSD_2(x_SSD)
        x_SSD = self.e2e_SSD_3(x_SSD, x_att[:, 14])
        x_SSD = self.e2e_SSD_4(x_SSD)
        x_SSD = self.e2n_SSD_1(x_SSD, x_att[:, 14])  
        x_SSD = self.e2n_SSD_2(x_SSD)

        x_SSC = self.e2e_SSC_1(x_SSC, x_att[:, 15])
        x_SSC = self.e2e_SSC_2(x_SSC)
        x_SSC = self.e2e_SSC_3(x_SSC, x_att[:, 15])
        x_SSC = self.e2e_SSC_4(x_SSC)
        x_SSC = self.e2n_SSC_1(x_SSC, x_att[:, 15])  
        x_SSC = self.e2n_SSC_2(x_SSC)

        '''length = 3 subnetwork'''
        x_DDCC = self.e2e_DDCC_1(x_DDCC, x_att[:, 16])
        x_DDCC = self.e2e_DDCC_2(x_DDCC)
        x_DDCC = self.e2e_DDCC_3(x_DDCC, x_att[:, 16])
        x_DDCC = self.e2e_DDCC_4(x_DDCC)
        x_DDCC = self.e2n_DDCC_1(x_DDCC, x_att[:, 16])  
        x_DDCC = self.e2n_DDCC_2(x_DDCC)

        x_DDSS = self.e2e_DDSS_1(x_DDSS, x_att[:, 17])
        x_DDSS = self.e2e_DDSS_2(x_DDSS)
        x_DDSS = self.e2e_DDSS_3(x_DDSS, x_att[:, 17])
        x_DDSS = self.e2e_DDSS_4(x_DDSS)
        x_DDSS = self.e2n_DDSS_1(x_DDSS, x_att[:, 17])  
        x_DDSS = self.e2n_DDSS_2(x_DDSS)

        x_CCSS = self.e2e_CCSS_1(x_CCSS, x_att[:, 18])
        x_CCSS = self.e2e_CCSS_2(x_CCSS)
        x_CCSS = self.e2e_CCSS_3(x_CCSS, x_att[:, 18])
        x_CCSS = self.e2e_CCSS_4(x_CCSS)
        x_CCSS = self.e2n_CCSS_1(x_CCSS, x_att[:, 18])  
        x_CCSS = self.e2n_CCSS_2(x_CCSS)

        x_DDCS = self.e2e_DDCS_1(x_DDCS, x_att[:, 19])
        x_DDCS = self.e2e_DDCS_2(x_DDCS)
        x_DDCS = self.e2e_DDCS_3(x_DDCS, x_att[:, 19])
        x_DDCS = self.e2e_DDCS_4(x_DDCS)
        x_DDCS = self.e2n_DDCS_1(x_DDCS, x_att[:, 19]) 
        x_DDCS = self.e2n_DDCS_2(x_DDCS)

        x_DDSC = self.e2e_DDSC_1(x_DDSC, x_att[:, 20])
        x_DDSC = self.e2e_DDSC_2(x_DDSC)
        x_DDSC = self.e2e_DDSC_3(x_DDSC, x_att[:, 20])
        x_DDSC = self.e2e_DDSC_4(x_DDSC)
        x_DDSC = self.e2n_DDSC_1(x_DDSC, x_att[:, 20])  
        x_DDSC = self.e2n_DDSC_2(x_DDSC)

        x_CCDS = self.e2e_CCDS_1(x_CCDS, x_att[:, 21])
        x_CCDS = self.e2e_CCDS_2(x_CCDS)
        x_CCDS = self.e2e_CCDS_3(x_CCDS, x_att[:, 21])
        x_CCDS = self.e2e_CCDS_4(x_CCDS)
        x_CCDS = self.e2n_CCDS_1(x_CCDS, x_att[:, 21])  
        x_CCDS = self.e2n_CCDS_2(x_CCDS)

        x_CCSD = self.e2e_CCSD_1(x_CCSD, x_att[:, 22])
        x_CCSD = self.e2e_CCSD_2(x_CCSD)
        x_CCSD = self.e2e_CCSD_3(x_CCSD, x_att[:, 22])
        x_CCSD = self.e2e_CCSD_4(x_CCSD)
        x_CCSD = self.e2n_CCSD_1(x_CCSD, x_att[:, 22])  
        x_CCSD = self.e2n_CCSD_2(x_CCSD)

        x_SSDC = self.e2e_SSDC_1(x_SSDC, x_att[:, 23])
        x_SSDC = self.e2e_SSDC_2(x_SSDC)
        x_SSDC = self.e2e_SSDC_3(x_SSDC, x_att[:, 23])
        x_SSDC = self.e2e_SSDC_4(x_SSDC)
        x_SSDC = self.e2n_SSDC_1(x_SSDC, x_att[:, 23])  
        x_SSDC = self.e2n_SSDC_2(x_SSDC)

        x_SSCD = self.e2e_SSCD_1(x_SSCD, x_att[:, 24])
        x_SSCD = self.e2e_SSCD_2(x_SSCD)
        x_SSCD = self.e2e_SSCD_3(x_SSCD, x_att[:, 24])
        x_SSCD = self.e2e_SSCD_4(x_SSCD)
        x_SSCD = self.e2n_SSCD_1(x_SSCD, x_att[:, 24])  
        x_SSCD = self.e2n_SSCD_2(x_SSCD)

        x_DSCD = self.e2e_DSCD_1(x_DSCD, x_att[:, 25])
        x_DSCD = self.e2e_DSCD_2(x_DSCD)
        x_DSCD = self.e2e_DSCD_3(x_DSCD, x_att[:, 25])
        x_DSCD = self.e2e_DSCD_4(x_DSCD)
        x_DSCD = self.e2n_DSCD_1(x_DSCD, x_att[:, 25])  
        x_DSCD = self.e2n_DSCD_2(x_DSCD)

        x_cmb = x.permute(0, 2, 3, 1)  
        x_DMN_cmb = x_DMN.permute(0, 2, 3, 1)
        x_CEN_cmb = x_CEN.permute(0, 2, 3, 1)
        x_SN_cmb = x_SN.permute(0, 2, 3, 1)
        x_DMN_CEN_cmb = x_DMN_CEN.permute(0, 2, 3, 1)
        x_DMN_SN_cmb = x_DMN_SN.permute(0, 2, 3, 1)
        x_CEN_SN_cmb = x_CEN_SN.permute(0, 2, 3, 1)
        x_DCS_cmb = x_DCS.permute(0, 2, 3, 1)
        x_DSC_cmb = x_DSC.permute(0, 2, 3, 1)
        x_CDS_cmb = x_CDS.permute(0, 2, 3, 1)
        x_DDC_cmb = x_DDC.permute(0, 2, 3, 1)
        x_DDS_cmb = x_DDS.permute(0, 2, 3, 1)
        x_CCD_cmb = x_CCD.permute(0, 2, 3, 1)
        x_CCS_cmb = x_CCS.permute(0, 2, 3, 1)
        x_SSD_cmb = x_SSD.permute(0, 2, 3, 1)
        x_SSC_cmb = x_SSC.permute(0, 2, 3, 1)
        x_DDCC_cmb = x_DDCC.permute(0, 2, 3, 1)
        x_DDSS_cmb = x_DDSS.permute(0, 2, 3, 1)
        x_CCSS_cmb = x_CCSS.permute(0, 2, 3, 1)
        x_DDCS_cmb = x_DDCS.permute(0, 2, 3, 1)
        x_DDSC_cmb = x_DDSC.permute(0, 2, 3, 1)
        x_CCDS_cmb = x_CCDS.permute(0, 2, 3, 1)
        x_CCSD_cmb = x_CCSD.permute(0, 2, 3, 1)
        x_SSDC_cmb = x_SSDC.permute(0, 2, 3, 1)
        x_SSCD_cmb = x_SSCD.permute(0, 2, 3, 1)
        x_DSCD_cmb = x_DSCD.permute(0, 2, 3, 1)

        X_concat = torch.cat((x_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DMN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_CEN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_SN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DMN_CEN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DMN_SN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_CEN_SN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DCS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DSC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_CDS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DDC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DDS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_CCD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_CCS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_SSD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_SSC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DDCC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DDSS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_CCSS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DDCS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DDSC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_CCDS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_CCSD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_SSDC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_SSCD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1),
                        x_DSCD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1)),
                       axis=1)

        y = self.self_attention_norm(X_concat)
        X_concat = self.self_attention_dropout(y)
        X_concat = X_concat + y
        X_concat = X_concat.reshape(X_concat.size(0) * X_concat.size(2), X_concat.size(1))
        X_trans = F.linear(X_concat, self.weight_attention, self.bias_attention)
        X_trans = X_trans.reshape(-1, x_DMN.size(1), x_DMN.size(2), 1)
        x_global = self.n2g_global(X_trans)
        x_final = x_global.reshape(x.size(0), -1)

        return x_final

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=8, temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        # self.kernel_size = kernel_size
        self.kernel_size1, self.kernel_size2 = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, self.kernel_size1, self.kernel_size2), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, att):  
        softmax_attention = self.attention(x)
        att = torch.unsqueeze(att, 1)
        att = att.repeat(1, self.K)
        softmax_attention = att * softmax_attention
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width) 
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
                                                                    self.in_planes // self.groups, self.kernel_size1,
                                                                    self.kernel_size2)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.d = input_shape[0]
        self.k = 4
        # self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        # self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.conv1xd = Dynamic_conv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=(self.d, 1), stride=1,
                                      padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.convdx1 = Dynamic_conv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=(1, self.d), stride=1,
                                      padding=0, dilation=1, groups=1, bias=True, K=self.k)

        self.nodes = 64

    def forward(self, A, att):
        A = A.view(-1, self.in_channel, self.nodes, self.nodes)

        a = self.conv1xd(A, att)
        b = self.convdx1(A, att)
        # a = self.conv1xd(A)
        # b = self.convdx1(A)

        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)

        return concat1 + concat2

class PreModel(nn.Module):
    def __init__(
            self,
            nodes: int,
            classes: int = 2,
    ):
        super(PreModel, self).__init__()
        self._classes = classes
        self._nodes = nodes
        self.dropout = 0.5
        self.encoder_su = Encoder(nodes=nodes)
        self.encoder_su2 = Encoder(nodes=nodes)
        self.cls_model_su = nn.Sequential(nn.Linear(self._nodes * self._nodes, self._nodes),
                                          nn.BatchNorm1d(self._nodes),
                                          nn.ReLU(),
                                          nn.Linear(self._nodes, self._classes),
                                          nn.BatchNorm1d(self._classes),
                                          nn.ReLU()
                                          )

        for layer in self.cls_model_su:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.cls_model_su2 = nn.Sequential(nn.Linear(self._nodes * self._nodes, self._nodes),
                                          nn.BatchNorm1d(self._nodes),
                                          nn.ReLU(),
                                          nn.Linear(self._nodes, self._classes),
                                          nn.BatchNorm1d(self._classes),
                                          nn.ReLU()
                                              )
        for layer in self.cls_model_su2:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.cls_loss = nn.CrossEntropyLoss()
        self.sim_loss = nn.MSELoss()
    def DiffLoss(self, input1, input2):

        batch_size = input1.size(0)
        input1_ = input1.view(batch_size, -1)
        input2_ = input2.view(batch_size, -1).detach()

        input1_l2_norm = torch.norm(input1_, p=2, dim=1, keepdim=True)
        input1_l2 = input1_.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2_, p=2, dim=1, keepdim=True)
        input2_l2 = input2_.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

    def co_training(self, logits_e2e2, logits_e2e, pseudo_labels_e2e2, pseudo_labels_e2e, high_confidence_mask_e2e2,
                         high_confidence_mask_e2e):
    
        hc_loss_e2e2 = self.cls_loss(logits_e2e2[high_confidence_mask_e2e], pseudo_labels_e2e[high_confidence_mask_e2e])
        hc_loss_e2e = self.cls_loss(logits_e2e[high_confidence_mask_e2e2], pseudo_labels_e2e2[high_confidence_mask_e2e2])

        lc_loss_e2e2 = self.cls_loss(logits_e2e2[~high_confidence_mask_e2e],pseudo_labels_e2e[~high_confidence_mask_e2e])
        lc_loss_e2e = self.cls_loss(logits_e2e[~high_confidence_mask_e2e2], pseudo_labels_e2e2[~high_confidence_mask_e2e2])

        hc_loss = hc_loss_e2e2 + hc_loss_e2e
        lc_loss = lc_loss_e2e2 + lc_loss_e2e
    
        total_loss = hc_loss

        return total_loss

    def forward(self, target_subnetwork_matrix, target_meta_path_matrices, target_feature, target_y_label, confidence):
        '''pretrained source supervision'''
        loss_sfda = self.sfda_co_training(target_subnetwork_matrix, target_meta_path_matrices, target_feature, target_y_label, confidence)
        return loss_sfda

    def compute_entropy(self,prob):
        entropy = - prob * torch.log(prob + 1e-9) - (1 - prob) * torch.log(1 - prob + 1e-9)
        return entropy
    def calculate_shot_loss(self, outputs, target_original_matrix, target_feature, epoch):
        total_loss = torch.tensor(0.0).to(device)

        if self.cls_par > 0:
            if epoch % 5 == 0 or self.pseudo_labels is None:
                self.update_pseudo_labels(target_original_matrix, target_feature)

            if self.pseudo_labels is not None:
                cls_loss = self.cls_par * self.cls_loss(outputs, self.pseudo_labels)
                total_loss += cls_loss

        if self.ent:
            softmax_out = nn.Softmax(dim=1)(outputs)
            entropy_loss = torch.mean(self.entropy_loss(softmax_out))

            if self.gent:
                msoftmax = softmax_out.mean(dim=0)
                gent_loss = -torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                entropy_loss -= gent_loss

            im_loss = entropy_loss * self.ent_par
            total_loss += im_loss

        return total_loss

    def contrastive_loss(self, feat1, feat2, temperature=0.5):
            feat1 = torch.nn.functional.normalize(feat1, dim=1)
            feat2 = torch.nn.functional.normalize(feat2, dim=1)

            logits = torch.matmul(feat1, feat2.T) / temperature

            labels = torch.arange(logits.shape[0]).to(logits.device)
            return torch.nn.functional.cross_entropy(logits, labels)


    def sfda_co_training(self, target_subnetwork_matrix, target_meta_path_matrices, target_feature, target_y_label,threshold=0.50):
        rep_e2e2 = self.encoder_su(target_subnetwork_matrix, target_meta_path_matrices)
        target_logits_e2e2 = self.cls_model_su2(rep_e2e2)

        rep_e2e = self.encoder_su2(target_subnetwork_matrix, target_meta_path_matrices)
        target_logits_e2e = self.cls_model_su(rep_e2e)

        target_probs_e2e2 = torch.softmax(target_logits_e2e2,dim=1)
        target_probs_e2e = torch.softmax(target_logits_e2e,dim=1)
        y_pseudo_e2e2 = target_probs_e2e2.max(1)[1]

        target_probs_e2e2_ = torch.clamp(target_probs_e2e2, min=1e-9, max=1.0)
        entrs_e2e2 = -torch.sum(target_probs_e2e2 * torch.log(target_probs_e2e2_ + 1e-9), dim=1)

        y_pseudo_e2e = target_probs_e2e.max(1)[1]
        target_probs_e2e_ = torch.clamp(target_probs_e2e, min=1e-9, max=1.0)
        entrs_e2e = -torch.sum(target_probs_e2e * torch.log(target_probs_e2e_ + 1e-9) , dim=1)

        #
        high_uncertainty_mask_e2e2 = entrs_e2e2 <= threshold
        high_uncertainty_mask_e2e = entrs_e2e <= threshold


        '''different loss'''
        # Compute the difference loss between E2E representations
        loss_diff = self.DiffLoss(rep_e2e2, rep_e2e)

        # Compute similarity loss between E2E logits
        loss_sim = self.sim_loss(target_logits_e2e2, target_logits_e2e) + self.sim_loss(target_logits_e2e, target_logits_e2e2)

        # Compute pseudo label consistency loss for low-entropy predictions
        loss_pse = self.co_training(target_logits_e2e2, target_logits_e2e, y_pseudo_e2e2, y_pseudo_e2e,
                                            high_uncertainty_mask_e2e2, high_uncertainty_mask_e2e)

        # Total loss with weighted consistency loss
        ''' compute entropy loss for T '''
        loss_entropy = torch.mean(entrs_e2e2) + torch.mean(entrs_e2e)
        loss_con = loss_sim + loss_pse

        '''total'''
        loss = loss_diff + loss_con + loss_entropy
        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
