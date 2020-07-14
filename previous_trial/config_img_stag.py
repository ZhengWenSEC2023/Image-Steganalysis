# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:39:36 2020

@author: Lenovo
"""

config = {"thres_bin_edges": 15,
          "save_dir_matFiles": r'/mnt/zhengwen/image_steganalysis/temp',
          "ori_image_dir": r'/mnt/zhengwen/image_steganalysis/train_img_with_prob',
          "stego_image_dir": r'/mnt/zhengwen/image_steganalysis/train_week5/stego_SUNI_025', # should be 0.05

          "stego_image_test_dir": r'/mnt/zhengwen/image_steganalysis/BOSSBase/S_UNIWARD_005', # should be 0.005

          "context_dir": r'original_resolution/context',
          "train_ori_context_name": r'train_context_4hop_new.h5',
          "weight_root_name": r'weight',
          "train_pos_target_name": r"pos_target.npy",
          "train_neg_target_name": r"neg_target.npy",
          "train_pos_contxt_name": r"pos_pixel_context.npy",
          "train_neg_contxt_name": r"neg_pixel_context.npy",
          'test_ori_contxt_name': r"test_context_4hop.h5",
          'test_contxt_name': "test_pixel_context.npy",
          'wind_rad': 2,
          'test_dir': r'/mnt/zhengwen/image_steganalysis/dataset/used_test',
          'regressor_dir': r'original_resolution/regressor',
          'regressor_name': r'Hier_random_forest_reg.pkl',
          'decision_thres': 0.5,
          'test_result_dir': r'original_resolution/pred_result',
          'result_name': r'25_train.h5',
          'plot_dir': r'original_resolution/plots',
          'one_hop_stp_lst_name': 'one_hop_stp_lst'
          }
