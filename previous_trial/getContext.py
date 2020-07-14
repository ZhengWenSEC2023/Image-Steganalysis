import numpy as np
from skimage import io
import os
from previous_trial.config_img_stag import config
import cv2

save_dir_matFiles = config['save_dir_matFiles']
window_rad = config['wind_rad']


def readImgToNumpy_pairs(ori_dir, stego_dir):
    """
    read paired images sequentially, return ori_imgs and stego_imgs,
    image names based on ori_dir
    """
    img_names = os.listdir(ori_dir)
    ori_imgs = []
    stego_imgs = []
    for img_name in img_names:
        ori_img = io.imread(os.path.join(ori_dir, img_name))
        stego_img = io.imread(os.path.join(stego_dir, img_name))
        ori_imgs.append(ori_img)
        stego_imgs.append(stego_img)
    ori_imgs = np.array(ori_imgs)
    stego_imgs = np.array(stego_imgs)
    return ori_imgs, stego_imgs


"""
test hop unchanged, 
the PixelHop_trans is changed, two output elements are deleted,
idx__stop_list is abandoned. 
Leaf_node_thres is changed.
"""


# def test_one_hop(ext_test_imgs):
#     test_one_hop_attri_saab = PixelHop_Unit(ext_test_imgs, dilate=1, window_size=window_rad,
#                                             idx_list=None, getK=False, pad='reflect',
#                                             weight_root=os.path.join(save_dir_matFiles, 'weight'),
#                                             Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
#                                             PCA_ener_percent=0.99,
#                                             useDC=False, stride=None, getcov=0, split_spec=0, hopidx=1)
#     test_one_hop_attri_biased = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
#                                                feature_ori=test_one_hop_attri_saab,
#                                                split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)
#     return test_one_hop_attri_biased
#
#
# def test_second_hop(test_one_hop_attri_response):
#     test_sec_hop_attri_saab = PixelHop_Unit(test_one_hop_attri_response, dilate=1, window_size=window_rad,
#                                             idx_list=None, getK=False, pad='reflect',
#                                             weight_root=os.path.join(save_dir_matFiles, 'weight'),
#                                             Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
#                                             PCA_ener_percent=0.97,
#                                             useDC=False, stride=None, getcov=0, split_spec=1, hopidx=2)
#     # print("sec hop neighborhood construction shape:", test_sec_hop_attri_saab.shape)
#     test_sec_hop_attri_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
#                                                  feature_ori=test_sec_hop_attri_saab,split_spec=1,
#                                                  hopidx=2, Pass_Ener_thrs=0.1)
#     # print("sec hop attri shape:", test_sec_hop_attri_response.shape)
#     return test_sec_hop_attri_response
#
#
# def test_third_hop(test_sec_hop_attri_response):
#     test_third_hop_attri_saab = PixelHop_Unit(test_sec_hop_attri_response, dilate=1, window_size=window_rad,
#                                               idx_list=None, getK=False, pad='reflect',
#                                               weight_root=os.path.join(save_dir_matFiles, 'weight'),
#                                               Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
#                                               PCA_ener_percent=0.97,
#                                               useDC=False, stride=None, getcov=0, split_spec=1, hopidx=3)
#     # print("Hop-3 feature ori shape:", test_third_hop_attri_saab.shape)
#     test_third_hop_attri_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
#                                                    feature_ori=test_third_hop_attri_saab,
#                                                    split_spec=1, hopidx=3, Pass_Ener_thrs=0.1)
#     # print("third hop attri shape:", test_third_hop_attri_response.shape)
#     return test_third_hop_attri_response
#
#
# def test_forth_hop(test_third_hop_attri_response):
#     test_fourth_hop_attri_saab = PixelHop_Unit(test_third_hop_attri_response, dilate=1, window_size=window_rad,
#                                                idx_list=None, getK=False, pad='reflect',
#                                                weight_root=os.path.join(save_dir_matFiles, 'weight'),
#                                                Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
#                                                PCA_ener_percent=0.97,
#                                                useDC=False, stride=None, getcov=0, split_spec=1, hopidx=4)
#     # print("Hop-3 feature ori shape:", test_third_hop_attri_saab.shape)
#     test_fourth_hop_attri_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
#                                                     feature_ori=test_fourth_hop_attri_saab,
#                                                     split_spec=1, hopidx=4, Pass_Ener_thrs=0.1)
#     # print("third hop attri shape:", test_third_hop_attri_response.shape)
#     return test_fourth_hop_attri_response


# def getContextFromImgs(imgs, weight_root):
#     """
#     get all the context for the subtree node
#     attribute: attribute of the latter part.
#     ori_imgs: input images, to get the size of context map.
#     """
#     one_res = test_one_hop(imgs)
#     sec_input = block_reduce(one_res, block_size=(1, 2, 2, 1), func=np.max)
#
#     fread = open(os.path.join(weight_root + str(1), 'idx_to_pass.pkl'), 'rb')
#     pass_idx = cPickle.load(fread)
#     fread.close()
#     all_nodes = np.arange(one_res.shape[-1]).tolist()
#     for i in pass_idx:
#         if i in all_nodes:
#             all_nodes.remove(i)
#     leaf_idx = all_nodes
#     one_res = one_res[:, :, :, leaf_idx]
#     print("Hop-1 attri shape:", one_res.shape)
#
#     sec_res = test_second_hop(sec_input)
#     thi_input = block_reduce(sec_res, block_size=(1, 2, 2, 1), func=np.max)
#
#     fread = open(os.path.join(weight_root + str(2), 'idx_to_pass.pkl'), 'rb')
#     pass_idx = cPickle.load(fread)
#     fread.close()
#     all_nodes = np.arange(sec_res.shape[-1]).tolist()
#     for i in pass_idx:
#         if i in all_nodes:
#             all_nodes.remove(i)
#     leaf_idx = all_nodes
#     sec_res = sec_res[:, :, :, leaf_idx]
#     print("Hop-2 attri shape:", sec_res.shape)
#
#     # Hop-3
#     thi_res = test_third_hop(thi_input)
#     for_input = block_reduce(thi_res, block_size=(1, 2, 2, 1), func=np.max)
#
#     fread = open(os.path.join(weight_root + str(3), 'idx_to_pass.pkl'), 'rb')
#     pass_idx = cPickle.load(fread)
#     fread.close()
#     all_nodes = np.arange(thi_res.shape[-1]).tolist()
#     for i in pass_idx:
#         if i in all_nodes:
#             all_nodes.remove(i)
#     leaf_idx = all_nodes
#     thi_res = thi_res[:, :, :, leaf_idx]
#     print("Hop-3 attri shape:", thi_res.shape)
#
#     # Hop-4
#     for_res = test_forth_hop(for_input)
#     print("Hop-4 attri shape:", for_res.shape)
#
#     return one_res, sec_res, thi_res, for_res
#
#
def getDataFromIthHop(attribute_each_img, hop_index, i, j):
    """
    get context data from all hop instead of padding
    hop_index begin with 1
    e.g. hop 1 with (512, 512)
         hop 2 with (256, 256)
         the idx of hop 2 will be divided by 2
    """
    factor = 2 ** hop_index - 1
    new_i = i // factor
    new_j = j // factor
    return attribute_each_img[new_i, new_j].copy()

def getContext(attribute, hop_idx):
    """
    get all the context for the subtree node
    attribute: attribute of the latter part.
    ori_imgs: input images, to get the size of context map.
    """
    factor = 2 ** (hop_idx - 1)
    S = attribute.shape
    context_later = np.zeros((S[0], S[-1], factor * S[1], factor * S[2]))
    attribute = np.moveaxis(attribute, -1, 1)
    new_S = (attribute.shape[0], attribute.shape[1], factor * attribute.shape[2], factor * attribute.shape[3])
    for k in range(context_later.shape[0]):
        for i in range(context_later.shape[1]):
            context_later[k][i] = cv2.resize(attribute[k][i],
                                             (new_S[2], new_S[3]),
                                             interpolation=cv2.INTER_NEAREST)
    context_later = np.moveaxis(context_later, 1, -1)
    return context_later

#
#
# def sobelImgs(gray_images):
#     gradients = []
#     for i in range(len(gray_images)):
#         gradients.append(sobel(gray_images[i]))
#     gradients = np.array(gradients)
#     return gradients
#
#
#
# def sobel(gray_image):
#     def gradNorm(grad):
#         return 255 * (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
#
#     m, n = gray_image.shape[0], gray_image.shape[1]
#     panel_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     panel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#     X = cv2.filter2D(gray_image, -1, panel_X)
#     Y = cv2.filter2D(gray_image, -1, panel_Y)
#     grad = np.sqrt((X ** 2) + (Y ** 2))
#     return gradNorm(grad).astype(int)
#

if __name__ == "__main__":
    """
    This program get context and selected location based on 
    high-rate stego images and original images.  
    """
    thres_bin_edges = config['thres_bin_edges']
    ori_image_dir = config['ori_image_dir']
    stego_image_dir = config['stego_image_dir']
    context_dir = config['context_dir']
    train_ori_context_name = config['train_ori_context_name']
    weight_root_name = config['weight_root_name']
    train_pos_target_name = config['train_pos_target_name']
    train_neg_target_name = config['train_neg_target_name']
    train_pos_contxt_name = config['train_pos_contxt_name']
    train_neg_contxt_name = config['train_neg_contxt_name']
    test_ori_contxt_name = config['test_ori_contxt_name']
    test_contxt_name = config['test_contxt_name']
    regressor_dir = config['regressor_dir']
    regressor_name = config['regressor_name']
    plot_dir = config['plot_dir']
    one_hop_stp_lst_name = config['one_hop_stp_lst_name']

    print("Begin preprocessing")
    ori_imgs, stego_imgs = readImgToNumpy_pairs(ori_image_dir, stego_image_dir)
    ori_imgs = ori_imgs[:, :, :, None]
    stego_imgs = stego_imgs[:, :, :, None]
    diffs = np.squeeze(ori_imgs.astype("double") - stego_imgs.astype("double"))

    hop1_context = np.load("low_freq_hop1.npy")
    hop2_context = np.load("high_low_freq_hop2.npy")
    hop2_context = getContext(hop2_context, 2)

    context = np.concatenate((hop1_context, hop2_context), axis=3)

    context_changed = context[np.where(diffs != 0)]

    unchanged_pos = np.where(diffs == 0)
    idx = np.random.permutation(unchanged_pos[0].shape[0])
    unchanged_pos = (unchanged_pos[0][idx][:len(context_changed)], unchanged_pos[1][idx][:len(context_changed)], unchanged_pos[2][idx][:len(context_changed)])
    context_unchanged = context[unchanged_pos]
    np.save("week5_context_unchanged.npy", context_unchanged)
    np.save("week5_context_changed.npy", context_changed)
    np.save("week5_pos_unchanged.npy", unchanged_pos)

    hop1_context_test = np.load("low_freq_hop1_test.npy")
    hop2_context_test = np.load("high_low_freq_hop2_test.npy")
    hop2_context_test = getContext(hop2_context_test, 2)
    context_test = np.concatenate((hop1_context_test, hop2_context_test), axis=3)
    context_test = np.reshape(context_test, (-1, context_test.shape[-1]))
    context_test = context_test[:5242880]
    np.save("week5_context_test.npy", context_test)




    # weight_root = os.path.join(save_dir_matFiles, weight_root_name)
    #
    #
    # ori_1_context, ori_2_context, ori_3_context, ori_4_context = getContextFromImgs(ori_imgs, weight_root)
    # stg_1_context, stg_2_context, stg_3_context, stg_4_context = getContextFromImgs(stego_imgs, weight_root)
    #
    # print("Finish preprocessing")
    #
    # print('Begin to select context')
    # selected_ori_context_diff = []
    # selected_stg_context_diff = []
    # diff_pos = []
    # selected_stego_target = []
    # count = 0
    # position_lists = [list(product(range(ori_imgs.shape[1]), range(ori_imgs.shape[2]))) for _ in range(ori_imgs.shape[0])]
    # position_changed_insequence = []
    # for k in range(len(ori_imgs)):
    #     print(k, str(1))
    #     diff = diffs[k]
    #     grad_ori = grad_oris[k]
    #     for i, j in product( range(diff.shape[0]), range(diff.shape[1]) ):
    #         if diff[i, j] != 0:
    #             ori_context = np.concatenate(
    #                 (getDataFromIthHop(ori_1_context[k], 1, i, j),
    #                  getDataFromIthHop(ori_2_context[k], 2, i, j),
    #                  getDataFromIthHop(ori_3_context[k], 3, i, j),
    #                  getDataFromIthHop(ori_4_context[k], 4, i, j)),
    #                 axis=0
    #             )
    #             stg_context = np.concatenate(
    #                 (getDataFromIthHop(stg_1_context[k], 1, i, j),
    #                  getDataFromIthHop(stg_2_context[k], 2, i, j),
    #                  getDataFromIthHop(stg_3_context[k], 3, i, j),
    #                  getDataFromIthHop(stg_4_context[k], 4, i, j)),
    #                 axis=0
    #             )
    #             selected_ori_context_diff.append(ori_context)
    #             selected_stg_context_diff.append(stg_context)
    #             selected_stego_target.append(grad_ori[i, j])
    #             diff_pos.append((k, i, j))          # record the changed location
    #             count += 1
    #             position_lists[k].remove((i, j))
    #             position_changed_insequence.append((k, i, j))
    # selected_ori_context_diff = np.array(selected_ori_context_diff)
    # selected_stg_context_diff = np.array(selected_stg_context_diff)
    # selected_stego_target = np.array(selected_stego_target)
    # # changed point for stego and cover
    # np.save(os.path.join(save_dir_matFiles, context_dir, 'selected_ori_context_diff.npy'), selected_ori_context_diff)
    # np.save(os.path.join(save_dir_matFiles, context_dir, 'selected_stg_context_diff.npy'), selected_stg_context_diff)
    # np.save(os.path.join(save_dir_matFiles, context_dir, 'selected_stego_target.npy'), selected_stego_target)
    # np.save(os.path.join(save_dir_matFiles, context_dir, 'diff_location.npy'), diff_pos)
    # print('Changed context successfully selected and saved')
    # print('Begin to randomly select unchanged context')
    #
    # selected_ori_unchanged_context = []
    # selected_stg_unchanged_context = []
    # selected_ori_target = []
    # same_pos = []
    # for k in range(len(ori_imgs)):
    #     print(k, str(2))
    #     diff = diffs[k]
    #     grad_ori = grad_oris[k]
    #     position_list = position_lists[k].copy()
    #     random.Random(0).shuffle(position_list)
    #     count_kth = count // len(ori_imgs)
    #     for i, j in position_list:
    #         if count_kth > 0:
    #             ori_context = np.concatenate(
    #                 (getDataFromIthHop(ori_1_context[k], 1, i, j),
    #                  getDataFromIthHop(ori_2_context[k], 2, i, j),
    #                  getDataFromIthHop(ori_3_context[k], 3, i, j),
    #                  getDataFromIthHop(ori_4_context[k], 4, i, j)),
    #                 axis=0
    #             )
    #             stg_context = np.concatenate(
    #                 (getDataFromIthHop(stg_1_context[k], 1, i, j),
    #                  getDataFromIthHop(stg_2_context[k], 2, i, j),
    #                  getDataFromIthHop(stg_3_context[k], 3, i, j),
    #                  getDataFromIthHop(stg_4_context[k], 4, i, j)),
    #                 axis=0
    #             )
    #             selected_ori_unchanged_context.append(ori_context)
    #             selected_stg_unchanged_context.append(stg_context)
    #             selected_ori_target.append(grad_ori[i, j])
    #             same_pos.append((k, i, j))
    #             count_kth -= 1
    # selected_ori_unchanged_context = np.array(selected_ori_unchanged_context)
    # selected_stg_unchanged_context = np.array(selected_stg_unchanged_context)
    # selected_ori_target = np.array(selected_ori_target)
    # # unchanged points for stego and cover
    # np.save(os.path.join(save_dir_matFiles, context_dir, 'selected_ori_unchanged_context.npy'), selected_ori_unchanged_context)
    # np.save(os.path.join(save_dir_matFiles, context_dir, 'selected_stg_unchanged_context.npy'), selected_stg_unchanged_context)
    # np.save(os.path.join(save_dir_matFiles, context_dir, 'selected_ori_target.npy'), selected_ori_target)
    # np.save(os.path.join(save_dir_matFiles, context_dir, 'same_location.npy'), same_pos)
    #
    # print('saved successfully')
    # #
    # """
    # Saab, feature extraction, all pixels
    # """
    #
    # """
    # calculate context
    # """
    #
    # # concatenate to be context
    # # context 1 is one_hop_attri
    # # context 2 is sec_hop_attri shape to 512*512
    # context_2 = getContext(sec_hop_attri, imgs)
    # context_3 = getContext(third_hop_attri, imgs)
    # context_4 = getContext(fourth_hop_attri, imgs)
    # context = np.concatenate((one_hop_attri, context_2, context_3, context_4), axis=-1)
    # # context finished
    #
    # # # write context to file
    # print("saving to files")
    # if not os.path.isdir(os.path.join(save_dir_matFiles, context_dir)):
    #     os.makedirs(os.path.join(save_dir_matFiles, context_dir))
    # h5f = h5py.File(os.path.join(save_dir_matFiles, context_dir, train_ori_context_name), 'w')
    # h5f.create_dataset('attri', data=context)
    # print("saved successfully")
    # # size_subset = 1000
    # # num_subset = int(context.shape[0] // size_subset + 1)
    # # for i in range(num_subset):
    # #     if np.size(context[i * size_subset: (i + 1) * size_subset]) != 0:
    # #         h5f.create_dataset('attribute' + str(i), data=context[i * size_subset: (i + 1) * size_subset])
    # h5f.close()
    #
    # # # read context from file
    # # context = []
    # # h5f = h5py.File(os.path.join(save_dir_matFiles, context_dir, train_ori_context_name), 'r')
    # # for i in range(len(list( h5f.keys() ))):
    # #     if isinstance(context, list):
    # #         context = h5f["attribute" + str(i)][:]
    # #     else:
    # #         context = np.concatenate( (context, h5f["attribute" + str(i)][:]), axis=0)
    # # h5f.close()
    # # print(context.shape)
    # end = time.time()
    # print("Time 1:", end - start)
    #
    # """
    # Select pixel features: positive
    # """
    # print(">>    SELECTING PIXEL FEATURES    <<")
    # print(context.shape)
    # pos_target = []
    # neg_target = []
    # pos_context = []
    # neg_context = []
    # counts_pos = np.zeros((edges.shape[0]), dtype=int)
    # counts_neg = np.zeros((edges.shape[0]), dtype=int)
    #
    # for k in range(edges.shape[0]):
    #     count = 0
    #     if labels[k] < 0:
    #         for i in range(edges.shape[1]):
    #             for j in range(edges.shape[2]):
    #                 if bin_edges[k, i, j] != 0:
    #                     neg_target.append(target_values[k, i, j])
    #                     neg_context.append(context[k, i, j])
    #                     count += 1
    #         counts_neg[k] = count
    #     else:
    #         for i in range(edges.shape[1]):
    #             for j in range(edges.shape[2]):
    #                 if bin_edges[k, i, j] != 0:
    #                     pos_target.append(target_values[k, i, j])
    #                     pos_context.append(context[k, i, j])
    #                     count += 1
    #         counts_pos[k] = count
    # print("total number of positive pixels:", counts_pos)
    # print("total number of negative pixels:", counts_neg)
    # print("pos target range:", np.max(pos_target), np.min(pos_target))
    # print("neg target range:", np.max(neg_target), np.min(neg_target))
    # print('saving target and context to files')
    # np.save(os.path.join(save_dir_matFiles, context_dir, train_pos_target_name), pos_target)
    # np.save(os.path.join(save_dir_matFiles, context_dir, train_neg_target_name), neg_target)
    # pos_context = np.array(pos_context)
    # neg_context = np.array(neg_context)
    # np.save(os.path.join(save_dir_matFiles, context_dir, train_pos_contxt_name), pos_context)
    # np.save(os.path.join(save_dir_matFiles, context_dir, train_neg_contxt_name), neg_context)
    # print('saved')
    # train_context = np.concatenate( (pos_context, neg_context), axis=0 )
    # train_target = np.concatenate( (pos_target, neg_target), axis=0 )

    # sum_counts_pos = sum(counts_pos)
    # sum_counts_neg = sum(counts_neg)
    # solve inbalancing, find equal and del, cost much time, CAREFUL
    # while sum_counts_pos > sum_counts_neg:
    #     for i in range(pos_context.shape[0]):
    #         for j in range(neg_context.shape[0]):
    #             if (pos_context[i] == neg_context[j]).all():
    #                 np.delete(pos_context, i, 0)
    #                 sum_counts_pos -= 1
    # while sum_counts_neg > sum_counts_pos:
    #     for i in range(neg_context.shape[0]):
    #         for j in range(pos_context.shape[0]):
    #             if (neg_context[i] == pos_context[j]).all():
    #                 np.delete(neg_context, i, 0)
    #                 sum_counts_neg -= 1

    # regr = RandomForestRegressor(n_estimators=200, random_state=0)
    # print('Begin to fit RF regressor')
    # regr.fit(train_context, train_target)
    # print('Fit finished')
    # if not os.path.isdir(os.path.join(save_dir_matFiles, regressor_dir)):
    #     os.makedirs(os.path.join(save_dir_matFiles, regressor_dir))
    # pkl = open(os.path.join(save_dir_matFiles, regressor_dir, regressor_name), 'wb')
    # pickle.dump(regr, pkl)
    # pkl.close()
    #
    # """
    # Val: percentage of positive class and negative class
    # """
    #
    # pos_result = regr.predict(pos_context)
    # neg_result = regr.predict(neg_context)
    #
    # pos_pos_percentage = np.sum(pos_result > 0) / np.sum(pos_result != 0)
    # pos_neg_percentage = np.sum(pos_result < 0) / np.sum(pos_result != 0)
    # neg_pos_percentage = np.sum(neg_result > 0) / np.sum(neg_result != 0)
    # neg_neg_percentage = np.sum(neg_result < 0) / np.sum(neg_result != 0)
    #
    # print("pos result", pos_pos_percentage, '%positive')
    # print("pos result", pos_neg_percentage, '%negative')
    # print("neg result", neg_pos_percentage, '%positive')
    # print("neg result", neg_neg_percentage, '%negative')
    #
    # if not os.path.isdir(os.path.join(save_dir_matFiles, plot_dir)):
    #     os.makedirs(os.path.join(save_dir_matFiles, plot_dir))
    # plt.hist(pos_result, bins=50, color=sns.desaturate("indianred", .8), alpha=.4)
    # plt.title("positive distribution")
    # plt.savefig(os.path.join(save_dir_matFiles, plot_dir, 'distribution_train_pos.png'))
    #
    # plt.hist(neg_result, bins=50, color=sns.desaturate("indianred", .8), alpha=.4)
    # plt.title("negative distribution")
    # plt.savefig(os.path.join(save_dir_matFiles, plot_dir, 'distribution_train_neg.png'))
    # """
    # Test
    # """
    # print("--------TEST PROCESS--------")
    #
    # test_dir = config['test_dir']
    # # read test images
    # test_filenames = np.random.permutation(os.listdir(test_dir)).tolist()
    # test_imgs = readImgToNumpy(test_dir)
    # test_edges = RobertsAlogrithm(test_imgs)
    # bin_test_edges = binarized(test_edges, thres=thres_bin_edges)
    # test_edges = cv2.normalize(test_edges, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    # print("test extended imgs shape:", test_imgs.shape)
    #
    # # %%
    # print(">>    FEARURE EXTRATION    <<")
    #
    # # extract context
    # # test hop-1
    # test_one_hop_attri_response = test_one_hop(test_imgs)
    # print(test_one_hop_attri_response.shape)  # 154, 256, 384, 12
    # test_one_hop_attri = test_one_hop_attri_response[:]  # 154, 256, 384, 8
    # test_sec_hop_input = block_reduce(test_one_hop_attri_response, block_size=(1, 2, 2, 1),
    #                                   func=np.max)  # 154, 128, 192, 12
    # # test Hop-2
    # test_sec_pass, test_sec_stay, test_sec_hop_attri_response = test_second_hop(test_sec_hop_input,
    #                                                                             idx_stop_list=[])
    # print(test_sec_pass.shape, test_sec_stay.shape, test_sec_hop_attri_response.shape)
    # test_sec_hop_attri = test_sec_stay
    # test_third_hop_input = block_reduce(test_sec_pass, block_size=(1, 2, 2, 1), func=np.max)
    # # test Hop-3
    # test_third_pass, test_third_stay, test_third_hop_attri_response = test_third_hop(test_third_hop_input,
    #                                                                                  idx_stop_list=[])
    # print(test_third_pass.shape, test_third_stay.shape, test_third_hop_attri_response.shape)
    # test_third_hop_attri = test_third_stay
    # test_forth_hop_input = block_reduce(test_third_pass, block_size=(1, 2, 2, 1), func=np.max)
    # # test Hop-4
    # test_forth_hop_attri_response = test_forth_hop(test_forth_hop_input, idx_stop_list=[])
    # test_forth_hop_attri = test_forth_hop_attri_response
    # print(test_forth_hop_attri.shape)
    #
    # # concatenate to be context
    # # test context 2
    # test_context_2 = getContext(test_sec_hop_attri, test_edges)
    # # test context 3
    # test_context_3 = getContext(test_third_hop_attri, test_edges)
    # # test context 4
    # test_context_4 = getContext(test_forth_hop_attri, test_edges)
    # test_context = np.concatenate((test_one_hop_attri, test_context_2, test_context_3, test_context_4), axis=-1)
    #
    # h5f = h5py.File(os.path.join(save_dir_matFiles, context_dir, test_ori_contxt_name), 'w')
    # size_subset = 1000
    # num_subset = int(test_context.shape[0] // size_subset + 1)
    # for i in range(num_subset):
    #     if np.size(test_context[i * size_subset: (i + 1) * size_subset]) != 0:
    #         h5f.create_dataset('attribute' + str(i), data=test_context[i * size_subset: (i + 1) * size_subset])
    # h5f.close()
    #
    # # test_context = []
    # # h5f = h5py.File( os.path.join(save_dir_matFiles, context_dir, test_ori_contxt_name), 'r')
    # # for i in range(len(list( h5f.keys() ))):
    # #     if isinstance(test_context, list):
    # #         test_context = h5f["attribute" + str(i)][:]
    # #     else:
    # #         test_context = np.concatenate( (test_context, h5f["attribute" + str(i)][:]), axis=0)
    # # h5f.close()
    # # print(test_context.shape)
    #
    # # %%  select positive & negative pixels of test imgs
    # selected_test_context = []
    # counts_test = np.zeros((test_edges.shape[0]), dtype=int)
    # for k in range(test_edges.shape[0]):
    #     count = 0
    #     for i in range(test_edges.shape[1]):
    #         for j in range(test_edges.shape[2]):
    #             if bin_edges[k, i, j] != 0:
    #                 selected_test_context.append(context[k, i, j])
    #                 count += 1
    #     counts_neg[k] = count
    # counts_test = np.array(counts_test)
    # np.save(os.path.join(save_dir_matFiles, context_dir, test_contxt_name), counts_test)