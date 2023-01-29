# # from scipy.io.wavfile import read
# # from fcwtold import *
# # import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# # from python_speech_features import delta
# # import timm
# # import pickle
# import numpy as np
# import seaborn as sb
# import pandas as pd
# from sklearn import preprocessing
# import fcwt
# import os
#
# # #Single input as test
# # test_wav = read("test.wav")
# # test_arr = np.array(test_wav[1], dtype='float32')
# #
# # #wavelet of choice
# # morl = fcwt.Morlet(2.0)
# #
# # #Sampling rate, freq. range start, freq. range end, frequency steps
# # fs = 16000
# # f0 = 1
# # f1 = 10000
# # fn = 227
# #
# # #Empty array for output
# # out = np.zeros((fn, len(test_arr)), dtype='csingle')
# # #Empty array for frequencies
# # freqs = np.zeros((fn), dtype='single')
# # #Get scales & freqs
# # scales = fcwt.Scales(morl, FCWT_LOGSCALES, fs, f0, f1, fn)
# # scales.getFrequencies(freqs)
# # #Initialize fCWT with morlet wddaxxvelet, 8 cores, and use optimizations
# # fcwt = FCWT(morl, 8, True)
# # #Perform fCWT using inputff, scales, and output array
# # fcwt.cwt(test_arr, scales, out)
# #
# # #Compute delta&deltadelta from output
# # #out_d = delta(out, 1)
# # #out_dd = delta(out_d, 1)
# #
# # emodb_images = np.load("emodb_fcwt_images_227_10000.npy")
# #
# # plt.imshow(np.absolute(emodb_images[0][:, :, 0]), aspect='auto')
# # plt.show
#
# emo_labels = np.load("EMO-DB_Audio&Labels/emodb_labels.npy")
# le_emo = preprocessing.LabelEncoder()
# le_emo.fit(emo_labels)
#
# ent_labels = np.load("ENTERFACE_MODELS/enterface05_label.npy")
# le_ent = preprocessing.LabelEncoder()
# le_ent.fit(ent_labels)
#
# emo_fcwt_rcs5_cm = np.load("EMODB_MODELS/EMO_FCWT_CV_CMS_RCS5_GNDBNO_WITH_CM.npy", allow_pickle=True)
# emo_stft_rcs5_cm = np.load("EMODB_MODELS/EMO_stft_CV_CMS_RCS5_GNDBNO_WITH_CM.npy", allow_pickle=True)
# emo_fcwt_GN_cm = np.load("EMODB_MODELS/EMO_FCWT_CV_CMS_RCS0_GNDByes_NEW_GNDB_TEST.npy", allow_pickle=True)
# emo_stft_GN_cm = np.load("EMODB_MODELS/EMO_stft_CV_CMS_RCS0_GNDByes_NEW_GNDB_TEST.npy", allow_pickle=True)
#
# ent_fcwt_rcs5_cm = np.load("ENTERFACE_MODELS/ENT_FCWT_CV_CMS_RCS5_GNDBNO_WITH_CM.npy", allow_pickle=True)
# ent_stft_rcs5_cm = np.load("ENTERFACE_MODELS/ENT_stft_CV_CMS_RCS5_GNDBNO_WITH_CM.npy", allow_pickle=True)
# ent_fcwt_GN_cm = np.load("ENTERFACE_MODELS/ENT_FCWT_CV_CMS_RCS0_GNDByes_NEW_GNDB_TEST.npy", allow_pickle=True)
# ent_stft_GN_cm = np.load("ENTERFACE_MODELS/ENT_stft_CV_CMS_RCS0_GNDByes_NEW_GNDB_TEST.npy", allow_pickle=True)
#
# ent_fcwt_sigma_cm = np.load("ENTERFACE_MODELS/ENT_FCWT_CV_CMS_RCS0_GNDBNO_WITH_CM_SIGMA_AS_DA.npy", allow_pickle=True)
# emo_fcwt_sigma_cm = np.load("EMODB_MODELS/EMO_FCWT_CV_CMS_RCS0_GNDBNO_WITH_CM_SIGMA_AS_DA.npy", allow_pickle=True)
#
# emo_fcwt_baseline_cm = np.load("EMODB_MODELS/EMO_FCWT_CV_CMS_RCS0_GNDBBASELINE.npy")
# emo_stft_baseline_cm = np.load("EMODB_MODELS/EMO_STFT_CV_CMS_RCS0_GNDBBASELINE.npy")
#
# ent_fcwt_baseline_cm = np.load("ENTERFACE_MODELS/ENT_FCWT_CV_CMS_RCS0_GNDBBASELINE.npy")
# ent_stft_baseline_cm = np.load("ENTERFACE_MODELS/ENT_STFT_CV_CMS_RCS0_GNDBNO_STFT_BASELINE.npy")
#
#
# def get_f1_recall_prec(cf_all, model):
#     precision_micro_all = []
#     precision_macro_all = []
#     recall_micro_all = []
#     recall_macro_all = []
#     f1_micro_all = []
#     f1_macro_all = []
#     for cf in cf_all:
#         TP = cf.diagonal()
#
#         precision_micro = TP.sum() / cf.sum()
#         recall_micro = TP.sum() / cf.sum()
#
#         precision_macro = np.nanmean(TP / cf.sum(0))
#         recall_macro = np.nanmean(TP / cf.sum(1))
#
#         f1_micro = (2 * (precision_micro * recall_micro)) / (precision_micro + recall_micro)
#         f1_macro = (2 * (precision_macro * recall_macro)) / (precision_macro + recall_macro)
#
#         precision_micro_all.append(precision_micro)
#         precision_macro_all.append(precision_macro)
#         recall_micro_all.append(recall_micro)
#         recall_macro_all.append(recall_macro)
#         f1_micro_all.append(f1_micro)
#         f1_macro_all.append(f1_macro)
#
#     avg_precision_micro = sum(precision_micro_all) / len(precision_micro_all)
#     avg_precision_macro = sum(precision_macro_all) / len(precision_macro_all)
#
#     avg_recall_micro = sum(recall_micro_all) / len(recall_micro_all)
#     avg_recall_macro = sum(recall_macro_all) / len(recall_macro_all)
#
#     avg_f1_micro = sum(f1_micro_all) / len(f1_micro_all)
#     avg_f1_macro = sum(f1_macro_all) / len(f1_macro_all)
#
#     print("Calculated for:", model)
#     # print("precision_micro:", avg_precision_micro)
#     print("precision_macro:", avg_precision_macro)
#     # print("recall_micro:", avg_recall_micro)
#     print("recall_macro:", avg_recall_macro)
#     # print("f1-micro:", avg_f1_micro)
#     print("f1-macro", avg_f1_macro)
#
#
# def get_cm(cms, le, save_name):
#     cms_a = np.sum(cms, axis=0)
#     cmn = (cms_a.astype('float') / cms_a.sum(axis=1)[:, np.newaxis]) * 100
#     fig, ax = plt.subplots(figsize=(10, 10))
#     sb.heatmap(cmn, annot=True, fmt='.2f', xticklabels=le.classes_, yticklabels=le.classes_)
#     plt.ylabel('Correct')
#     plt.xlabel('Predicted')
#     plt.savefig(save_name)
#     return plt
#
#
# # get_f1_recall_prec(emo_fcwt_rcs5_cm, "EMODB-FCWT-RCS5")
# # get_f1_recall_prec(emo_stft_rcs5_cm, "EMODB-stft-RCS5")
# # get_f1_recall_prec(emo_fcwt_GN_cm, "EMODB-FCWT-GN")
# # get_f1_recall_prec(emo_stft_rcs5_cm, "EMODB-stft-GN")
# # get_f1_recall_prec(emo_fcwt_sigma_cm, "EMODB-FCWT-sigma")
# #
# # get_f1_recall_prec(ent_fcwt_rcs5_cm, "ent-FCWT-RCS5")
# # get_f1_recall_prec(ent_stft_rcs5_cm, "ent-stft-RCS5")
# # get_f1_recall_prec(ent_fcwt_GN_cm, "ent-FCWT-GN")
# # get_f1_recall_prec(ent_stft_rcs5_cm, "ent-stft-GN")
# #
# # get_f1_recall_prec(ent_fcwt_sigma_cm, "ent-FCWT-sigma")
# # get_f1_recall_prec(emo_fcwt_sigma_cm, "emo-FCWT-sigma")
# #
# # get_f1_recall_prec(emo_fcwt_baseline_cm, "emo-FCWT-baseline")
# # get_f1_recall_prec(emo_stft_baseline_cm, "emo-stft-baseline")
# #
# # get_f1_recall_prec(ent_fcwt_baseline_cm, "ent-FCWT-baseline")
# # get_f1_recall_prec(ent_stft_baseline_cm, "ent-stft-baseline")
#
# # get_cm(emo_fcwt_rcs5_cm, le_emo, "EMODB-FCWT-RCS5-CM.png")
# # get_cm(emo_stft_rcs5_cm, le_emo,"EMODB-stft-RCS5-CM.png")
# # get_cm(emo_fcwt_GN_cm, le_emo,"EMODB-FCWT-GN-CM.png")
# # get_cm(emo_stft_rcs5_cm, le_emo,"EMODB-stft-GN-CM.png")
# # get_cm(emo_fcwt_sigma_cm, le_emo,"EMODB-FCWT-sigma-CM.png")
# #
# # get_cm(ent_fcwt_rcs5_cm, le_ent,"ent-FCWT-RCS5-CM.png")
# # get_cm(ent_stft_rcs5_cm, le_ent,"ent-stft-RCS5-CM.png")
# # get_cm(ent_fcwt_GN_cm, le_ent,"ent-FCWT-GN-CM.png")
# # get_cm(ent_stft_rcs5_cm, le_ent,"ent-stft-GN-CM.png")
# #
# # get_cm(ent_fcwt_sigma_cm, le_ent,"ent-FCWT-sigma-CM.png")
# # get_cm(emo_fcwt_sigma_cm, le_emo,"emo-FCWT-sigma-CM.png")
# #
# # get_cm(emo_fcwt_baseline_cm, le_emo,"emo-FCWT-baseline-CM.png")
# # get_cm(emo_stft_baseline_cm, le_emo,"emo-stft-baseline-CM.png")
# # get_cm(ent_fcwt_baseline_cm, le_ent,"ent-FCWT-baseline-CM.png")
# # get_cm(ent_stft_baseline_cm, le_ent,"ent-stft-baseline-CM.png")
#
#
#
#
#
#
#
# # sigma_range = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
# # sigma_range_labels = ["1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
# #
# # sigma_range_accs_emo = np.load("EMODB_MODELS/sigma_accuries22-1.npy", allow_pickle=True)
# # sigma_range_accs_ent = np.load("ENTERFACE_MODELS/ent_sigma_accs.npy", allow_pickle=True)
# #
# #
# # sb.set(style="darkgrid")
# # plot = sb.lineplot(sigma_range_accs_emo, label="EMO-DB")
# # sb.lineplot(sigma_range_accs_ent, label="eNTERFACE05")
# # sb.lineplot()
# # plot.set(xticks=sigma_range)
# # plot.set(xticklabels=sigma_range_labels)
# # plot.set(xlabel="Sigma value")
# # plot.set(ylabel="Accuracy")
# # plt.tight_layout()
# #
# # # ax = plt.gca()
# # # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
# # plt.savefig("sigma_accuracies.png", dpi=300)
# # plt.show()
#
# train_accs_EMO_fcwt_RCS = np.load("EMODB_MODELS/EMO_FCWT_CV_ACCS_RCS5_GNDBNO_WITH_CM.npy", allow_pickle=True)
# train_accs_EMO_stft_RCS = np.load("EMODB_MODELS/EMO_stft_CV_ACCS_RCS5_GNDBNO_WITH_CM.npy", allow_pickle=True)
# train_accs_EMO_fcwt_GN = np.load("EMODB_MODELS/EMO_FCWT_CV_ACCS_RCS0_GNDBYES_WITH_CM.npy", allow_pickle=True)
# train_accs_EMO_stft_GN = np.load("EMODB_MODELS/EMO_STFT_CV_ACCS_RCS0_GNDBYES_WITH_CM.npy", allow_pickle=True)
#
#
# plt.style.use('seaborn-v0_8-darkgrid')
# X = np.arange(10)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.bar(X - 0.25, train_accs_EMO_fcwt_RCS, width = 0.20, align='center', edgecolor='black', linewidth=1)
# ax.bar(X - 0.05, train_accs_EMO_stft_RCS, width = 0.20, align='center', edgecolor='black', linewidth=1)
# ax.bar(X + 0.15, train_accs_EMO_fcwt_GN, width = 0.20, align='center', edgecolor='black', linewidth=1)
# ax.bar(X + 0.35, train_accs_EMO_stft_GN, width = 0.20, align='center', edgecolor='black', linewidth=1)
#
# ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# tick_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
# plt.ylim(0.5, 0.9)
# plt.xticks(ticks, tick_labels)
# plt.xlabel("CV-fold")
# plt.ylabel("Accuracy")
# ax.legend(['fCWT-RCS', 'STFT-RCS', 'fCWT-WGN', 'STFT-WGN'], prop={'size': 9})
# plt.savefig("cv-accs.png")
# plt.show()
#
#
# # df = pd.DataFrame(columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
# # df.loc[len(df)] = train_accs_EMO_fcwt_RCS
# # df.loc[len(df)] = train_accs_EMO_stft_RCS
# # df.loc[len(df)] = train_accs_EMO_fcwt_GN
# # df.loc[len(df)] = train_accs_EMO_stft_GN
# # df.index = ["fCWT-RCS", "STFT-RCS", "fCWT-WGN", "STFT-WGN"]
#
#
# # sb.set(style="darkgrid")
# # plot = sb.barplot(x = df.index, y=df.values)
# # # sb.barplot(train_accs_EMO_stft, label="EMO-STFT")
# # # sb.barplot(train_accs_ent_fcwt, label="ENT05-FCWT")
# # # sb.barplot(train_accs_ent_stft, label="ENT05-STFT")
# # plot.set(xlabel="CV-fold")
# # plot.set(ylabel="Accuracy")
# # plt.tight_layout()
# # plt.savefig("cv-accs.png", dpi=300)
# # plt.show()
#
