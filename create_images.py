import data_augmentation
import create_TF_images as TF

import numpy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import os

def generate_fcwt_stft_images(audio, labels, out_size, fs, f0, f1, fn, mor_sig, save, to_int, gn_db, rcs, gni, emo, multiple_sigma):
    #Load audio and labels
    emodb_audio = TF.pcm2float(audio)
    emodb_labels = labels
    #generate fcwt images
    #fcwt_images_ori = TF.generate_fcwt_images(np.array(emodb_audio), out_size, fs, f0, f1, fn, mor_sig, save, to_int)
    # generate STFT images
    if emo:
        hop = (3 * 16000) // out_size
        stft_images_ori = TF.generate_stft_images(
            np.array(emodb_audio), out_size=out_size, signal_len=3, Fs=16000, n_fft=455, win_length=455, hop_length=hop,
            toint=True, save=False)
    else:
        hop = (3 * 44100) // out_size
        stft_images_ori = TF.generate_stft_images(
            np.array(emodb_audio), out_size=out_size, signal_len=3, Fs=44100, n_fft=1023, win_length=1023, hop_length=hop,
            toint=True, save=False)
    # RCS fCWT
    #fcwt_images_all_RCS, fcwt_labels_all_RCS = data_augmentation.random_shift_data(fcwt_images_ori, emodb_labels, rcs)
    # RCS STFT
    stft_images_all_RCS, stft_labels_all_RCS = data_augmentation.random_shift_data(stft_images_ori, emodb_labels, rcs)
    #If noise is added, perform everything for this proportion as well
    # if len(multiple_sigma) > 0:
    #     for sigma in multiple_sigma:
    #         print("adding sigma %s" % sigma)
    #         fcwt_extra_sigma = TF.generate_fcwt_images(np.array(emodb_audio), out_size, fs, f0, f1, fn, sigma, save, to_int)
    #         fcwt_extra_sigma_labels = emodb_labels
    #         fcwt_images_all_RCS = np.concatenate((fcwt_images_all_RCS, fcwt_extra_sigma), axis=0)
    #         fcwt_labels_all_RCS = np.concatenate((fcwt_labels_all_RCS, fcwt_extra_sigma_labels), axis=0)
    if len(gn_db) > 0:
        # Generate gni number of gn_db levels in a linear space between 5 and 30. (gni = gaussion noise iterations).
        for db in gn_db:
            print("adding noise for following level of gaussion noise")
            print(db)
            audio_GN, labels_GN = data_augmentation.audio_augmentation(emodb_audio, emodb_labels, fs, db)
            fcwt_images_GN = TF.generate_fcwt_images(audio_GN, out_size, fs, f0, f1, fn, mor_sig, save, to_int)
            if emo:
                stft_images_GN = TF.generate_stft_images(audio_GN, out_size=out_size, signal_len=3, Fs=16000, n_fft=455,
                                                         win_length=455, hop_length=hop, toint=True, save=False)
            else:
                stft_images_GN = TF.generate_stft_images(audio_GN, out_size=out_size, signal_len=3, Fs=44100, n_fft=1023,
                                                         win_length=1023, hop_length=hop, toint=True, save=False)
            #fcwt_images_GN_RCS, fcwt_labels_GN_RCS = data_augmentation.random_shift_data(fcwt_images_GN, labels_GN, rcs)
            stft_images_GN_RCS, stft_labels_GN_RCS = data_augmentation.random_shift_data(stft_images_GN, labels_GN, rcs)
            # Concatenate for training and save
            #fcwt_images_all_RCS = np.concatenate((fcwt_images_all_RCS, fcwt_images_GN_RCS), axis=0)
            #fcwt_labels_all_RCS = np.concatenate((fcwt_labels_all_RCS, fcwt_labels_GN_RCS), axis=0)
            stft_images_all_RCS = np.concatenate((stft_images_all_RCS, stft_images_GN_RCS), axis=0)
            stft_labels_all_RCS = np.concatenate((stft_labels_all_RCS, stft_labels_GN_RCS), axis=0)
    if save:
        newpath = '%s\emodb_images_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (os.getcwd(), f1, fn, mor_sig, gn_db, p, rcs)
        if not os.path.exists(newpath):
             os.makedirs(newpath)
        np.save(arr=fcwt_images_all_RCS,
                 file='%s\emodb05_fcwt_images_all_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (newpath, f1, fn, mor_sig, gn_db, p, rcs))
        np.save(arr=fcwt_labels_all_RCS,
                 file='%s\emodb05_fcwt_labels_all_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (newpath, f1, fn, mor_sig, gn_db, p, rcs))
        np.save(arr=stft_images_all_RCS,
                 file='%s\emodb05_stft_images_all_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (newpath, f1, fn, mor_sig, gn_db, p, rcs))
        np.save(arr=stft_labels_all_RCS,
                 file='%s\emodb05_stft_labels_all_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (newpath, f1, fn, mor_sig, gn_db, p, rcs))
    return stft_images_all_RCS, stft_labels_all_RCS

def generate_fcwt_stft_images_emodb_sf(audio_path, out_size, fs, f0, f1, fn, sigma_values, save, to_int, gn_db, rcs, p, gni):
    #Load audio and labels
    emodb_audio = TF.pcm2float(np.load(audio_path, allow_pickle=True))
    emodb_labels = np.load('EMO-DB_Audio&Labels/emodb_labels.npy', allow_pickle=True)
    #generate fcwt images
    fcwt_images_ori = TF.generate_fcwt_images_sf(np.array(emodb_audio), out_size, fs, f0, f1, fn, sigma_values, save, to_int)
    # generate STFT images
    hop = (3 * fs) // out_size
    stft_images_ori = TF.generate_stft_images(
        np.array(emodb_audio), out_size=out_size, signal_len=3, Fs=fs, n_fft=455, win_length=455, hop_length=hop,
        toint=True, save=False)
    # RCS fCWT
    fcwt_images_all_RCS, fcwt_labels_all_RCS = data_augmentation.random_shift_data(fcwt_images_ori, emodb_labels, rcs)
    # RCS STFT
    stft_images_all_RCS, stft_labels_all_RCS = data_augmentation.random_shift_data(stft_images_ori, emodb_labels, rcs)
    #If noise is added, perform everything for this proportion as well
    if p > 0:
        # Generate gni number of gn_db levels in a linear space between 5 and 30. (gni = gaussion noise iterations).
        gn_db = np.linspace(5, 30, gni).astype(int)
        for db in gn_db:
            print("adding noise for following levels of gaussion noise")
            print(gn_db)
            audio_GN, labels_GN = data_augmentation.audio_augmentation(emodb_audio, emodb_labels, fs, db, p)
            fcwt_images_GN = TF.generate_fcwt_images(audio_GN, out_size, fs, f0, f1, fn, mor_sig, save, to_int)
            stft_images_GN = TF.generate_stft_images(audio_GN, out_size=out_size, signal_len=3, Fs=fs, n_fft=455,
                                                     win_length=455, hop_length=hop, toint=True, save=False)
            fcwt_images_GN_RCS, fcwt_labels_GN_RCS = data_augmentation.random_shift_data(fcwt_images_GN, labels_GN, rcs)
            stft_images_GN_RCS, stft_labels_GN_RCS = data_augmentation.random_shift_data(stft_images_GN, labels_GN, rcs)
            # Concatenate for training and save
            fcwt_images_all_RCS = np.concatenate((fcwt_images_all_RCS, fcwt_images_GN_RCS), axis=0)
            fcwt_labels_all_RCS = np.concatenate((fcwt_labels_all_RCS, fcwt_labels_GN_RCS), axis=0)
            stft_images_all_RCS = np.concatenate((stft_images_all_RCS, stft_images_GN_RCS), axis=0)
            stft_labels_all_RCS = np.concatenate((stft_labels_all_RCS, stft_labels_GN_RCS), axis=0)
    if save:
        newpath = '%s\emodb_images_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (os.getcwd(), f1, fn, mor_sig, gn_db, p, rcs)
        if not os.path.exists(newpath):
             os.makedirs(newpath)
        np.save(arr=fcwt_images_all_RCS,
                 file='%s\emodb05_fcwt_images_all_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (newpath, f1, fn, mor_sig, gn_db, p, rcs))
        np.save(arr=fcwt_labels_all_RCS,
                 file='%s\emodb05_fcwt_labels_all_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (newpath, f1, fn, mor_sig, gn_db, p, rcs))
        np.save(arr=stft_images_all_RCS,
                 file='%s\emodb05_stft_images_all_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (newpath, f1, fn, mor_sig, gn_db, p, rcs))
        np.save(arr=stft_labels_all_RCS,
                 file='%s\emodb05_stft_labels_all_f%s_fn%s_sig%s_GN%s_P%s_RCS%s' % (newpath, f1, fn, mor_sig, gn_db, p, rcs))
    return fcwt_images_all_RCS, stft_images_all_RCS, fcwt_labels_all_RCS

