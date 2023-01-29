import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from fine_tune import generate_dcnn_input, create_dataloader, create_model, emotions_to_label, train_model
import create_images


def train_model_fcwt_stft(X_train_fcwt, y_train_fcwt, X_val_fcwt,
                          y_val_fcwt, X_train_stft, y_train_stft, X_val_stft,
                          y_val_stft,
                          le, batch_size, learning_rate, momentum, n_epochs, save):
    torch.manual_seed(0)
    # create dataloaders
    dataloader_train_fcwt = create_dataloader(x=X_train_fcwt, y=y_train_fcwt, mode="train", save=False,
                                              path="dataloader_train16_shift", batch_size=batch_size)
    dataloader_val_fcwt = create_dataloader(x=X_val_fcwt, y=y_val_fcwt, mode="val", save=False,
                                            path="dataloader_val16_shift", batch_size=batch_size)
    dataloaders_fcwt = {"train": dataloader_train_fcwt, "val": dataloader_val_fcwt}
    dataloader_train_stft = create_dataloader(x=X_train_stft, y=y_train_stft, mode="train", save=False,
                                              path="dataloader_train16_shift", batch_size=batch_size)
    dataloader_val_stft = create_dataloader(x=X_val_stft, y=y_val_stft, mode="val", save=False,
                                            path="dataloader_val16_shift", batch_size=batch_size)
    dataloaders_stft = {"train": dataloader_train_stft, "val": dataloader_val_stft}

    # create models
    model_fcwt = create_model(model_name='alexnet', n_classes=len(le.classes_), pretrained=True)
    model_stft = create_model(model_name='alexnet', n_classes=len(le.classes_), pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer_fcwt = optim.SGD(model_fcwt.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_stft = optim.SGD(model_stft.parameters(), lr=learning_rate, momentum=momentum)

    # train models (uncomment bottom line if you also want to train STFT model. Usually not needed as paramters for STFT
    # are set and thus only needs to be trained once.
    print("n_epochs={}, batch_size={},lr={},momentum={}".format(n_epochs, batch_size, learning_rate, momentum))
    print("Starting fcwt model training \n", '-' * 20)
    best_model_fcwt, fcwt_history, time_elapsed = train_model(model_fcwt, dataloaders_fcwt, criterion, optimizer_fcwt,
                                                             n_epochs)
    print("Starting stft model training \n", '-' * 20)
    best_model_stft, stft_history, time_elapsed = train_model(model_stft, dataloaders_stft, criterion, optimizer_stft,
                                                             n_epochs)

    # Save models if you want
    if save:
        save_path = "fcwt_model"
        torch.save(best_model_fcwt.state_dict(), save_path)
        np.save(arr=torch.tensor(fcwt_history).detach().cpu().numpy(), file="%s_history" % save_path)
        print("SAVED as %s and %s_history" % (save_path, save_path))

        save_path = "stft_model"
        torch.save(best_model_stft.state_dict(), save_path)
        np.save(arr=torch.tensor(stft_history).detach().cpu().numpy(), file="%s_history" % save_path)
        print("SAVED as %s and %s_history" % (save_path, save_path))

    # Return models (add best_model_stft after best_model_fcwt if you want to return this as well)
    return best_model_fcwt, best_model_stft


def get_split(cross_val_nr, emo):
    if emo:
        speaker_index = [[0, 48], [49, 106], [107, 149], [150, 187], [188, 242], [243, 277], [278, 338], [339, 407],
                         [408, 463], [464, 534]]
        images = np.load("EMO-DB_Audio&Labels/emodb_audio.npy", allow_pickle=True)
        labels = np.load("EMO-DB_Audio&Labels/emodb_labels.npy", allow_pickle=True)
    else:
        speaker_index = [[0, 149], [150, 299], [300, 449], [450, 596], [597, 746], [747, 896], [897, 1046],
                         [1047, 1196], [1197, 1286]]
        images = np.load("enterface05_audio_wo-subject6.pkl", allow_pickle=True)
        labels = np.load("ENTERFACE_MODELS/enterface05_label.npy")
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    output = open('label_encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    if cross_val_nr < len(speaker_index) - 1:
        index_start_test = speaker_index[cross_val_nr][0]
        index_end_test = speaker_index[cross_val_nr][1]
        index_start_val = speaker_index[cross_val_nr + 1][0]
        index_end_val = speaker_index[cross_val_nr + 1][1]
        index_list = list(range(index_start_test, (index_end_val + 1)))

        X_test = images[index_start_test:(index_end_test + 1)]
        y_test = labels[index_start_test:(index_end_test + 1)]
        X_val = images[index_start_val:(index_end_val + 1)]
        y_val = labels[index_start_val:(index_end_val + 1)]

        X_train = np.delete(images, index_list, axis=0)
        y_train = np.delete(labels, index_list, axis=0)
    else:
        index_start_test = speaker_index[cross_val_nr][0]
        index_end_test = speaker_index[cross_val_nr][1]
        index_start_val = speaker_index[0][0]
        index_end_val = speaker_index[0][1]
        index_list_test = list(range(index_start_test, (index_end_test + 1)))
        index_list_val = list(range(index_start_val, (index_end_val + 1)))
        index_list = index_list_test + index_list_val

        X_test = images[index_start_test:(index_end_test + 1)]
        y_test = labels[index_start_test:(index_end_test + 1)]
        X_val = images[index_start_val:(index_end_val + 1)]
        y_val = labels[index_start_val:(index_end_val + 1)]
        X_train = np.delete(images, index_list, axis=0)
        y_train = np.delete(labels, index_list, axis=0)
    return X_train, y_train, X_test, y_test, X_val, y_val, le


def start_training_pipeline(X_train_fcwt, y_train_fcwt, X_test_fcwt, y_test_fcwt, X_val_fcwt, y_val_fcwt,
                            X_train_stft, y_train_stft, X_test_stft, y_test_stft, X_val_stft, y_val_stft,
                            le, mor_sig, gn_db, rcs, gni, fs, f0, f1, fn, emo, save=False):
    # Train models
    model_fcwt, model_stft = train_model_fcwt_stft(X_train_fcwt, y_train_fcwt, X_val_fcwt, y_val_fcwt,
                                                   X_train_stft, y_train_stft, X_val_stft, y_val_stft,
                                                   le, batch_size=16, learning_rate=0.001, momentum=0.9, n_epochs=60,
                                                   save=False)

    # load best model state into evaluation model (add model_stft after model_fcwt if you want to evaluate it
    m_fcwt = create_model('alexnet', len(le.classes_), pretrained=True)
    m_fcwt.load_state_dict(model_fcwt.state_dict())
    m_fcwt.eval()

    # uncomment below if you want to evaluate fcwt model against stft model
    m_stft = create_model('alexnet', len(le.classes_), pretrained=True)
    m_stft.load_state_dict(model_stft.state_dict())
    m_stft.eval()

    # Convert test set into correct alexnet input
    xtest_fcwt = generate_dcnn_input(X_test_fcwt, mode='test')
    xtest_stft = generate_dcnn_input(X_test_stft, mode='test')

    # Make predictions on test set using models
    y_pred_fcwt = m_fcwt(xtest_fcwt)
    y_pred_stft = m_stft(xtest_stft)

    # Given that model outputs 'confidence score' for each output category, determine highest score in prediction array
    # and convert to prediction
    pred_fcwt = []
    pred_stft = []
    for i in range(len(y_pred_stft)):
        pred_fcwt.append(torch.argmax(y_pred_fcwt[i]).numpy())
        pred_stft.append(torch.argmax(y_pred_stft[i]).numpy())

    # Determine accuracy of model on test set
    acc_fcwt = accuracy_score(y_test_fcwt, pred_fcwt)
    acc_stft = accuracy_score(y_test_stft, pred_stft)
    cm_fcwt = confusion_matrix(y_test_fcwt, pred_fcwt)
    cm_stft = confusion_matrix(y_test_stft, pred_stft)
    print("fcwt model acc: %s" % acc_fcwt)
    print("stft model acc: %s" % acc_stft)

    if save:
        #Save fcwt model in sigma_range_test_emodb folder
        save_path_fcwt = 'EMODB_MODELS\fCWT_RCS%s_SIG%s_GNDB0_acc%s' % (rcs, mor_sig, acc_fcwt)
        save_path_stft = 'EMODB_MODELS\STFT_RCS%s_SIG%s_GNDB0_acc%s' % (rcs, mor_sig, acc_fcwt)
        np.save(arr=y_pred_fcwt, file='EMODB_MODELS\fCWT_RCS%s_SIG%s_GNDB0_acc%s_predictions')
        np.save(arr=test_labels, file='EMODB_MODELS\fCWT_RCS%s_SIG%s_GNDB0_acc%s_test_labels')
        np.save(arr=y_pred_stft, file='EMODB_MODELS\stft_RCS%s_SIG%s_GNDB0_acc%s_predictions')
        np.save(arr=test_labels, file='EMODB_MODELS\stft_RCS%s_SIG%s_GNDB0_acc%s_test_labels')
        torch.save(model_fcwt.state_dict(), save_path_fcwt)
        torch.save(model_stft.state_dict(), save_path_stft)

    return m_fcwt, acc_fcwt, cm_fcwt, m_fcwt, acc_stft, cm_stft


############## Full training pipeline below #############################
# parameters
mor_sig = 0
gn_db = []
rcs = 0
p = 0
gni = 0
sigma_range = np.linspace(1, 100, 50)


def sigma_range_test(sigma_range, fs, f0, f1, fn, emo, save=True):
    if emo:
        classes = 7
        nr_of_groups = 10
        audio_images = np.load("EMO-DB_Audio&Labels/emodb_audio.npy", allow_pickle=True)
        audio_labels = np.load("EMO-DB_Audio&Labels/emodb_labels.npy", allow_pickle=True)
    else:
        classes = 6
        nr_of_groups = 9
        audio_images = np.load("enterface05_audio_wo-subject6.pkl", allow_pickle=True)
        audio_labels = np.load("ENTERFACE_MODELS/enterface05_label.npy", allow_pickle=True)
    best_sigma = 0
    best_acc_sigma = 0
    sigma_accuracies = []
    best_sigma_model = create_model('alexnet', classes, pretrained=True)
    for sigma in sigma_range:
        cv_best_acc = 0
        cv_all_acc = []
        best_cv_model = create_model('alexnet', classes, pretrained=True)
        print("Generating all images for sigma: %s" % sigma)
        fcwt_images, stft_images, labels = create_images.generate_fcwt_stft_images(audio_images, audio_labels,
                                                                                   out_size=227, fs=fs, f0=f0, f1=f1,
                                                                                   fn=fn,
                                                                                   mor_sig=sigma, gn_db=gn_db, rcs=rcs,
                                                                                   gni=gni, emo=emo,
                                                                                   save=False, to_int=False)
        # Cross validation of one sigma
        for cross_val_nr in range(nr_of_groups):
            print("Testing sigma %s, crossval %s/10" % (sigma, cross_val_nr + 1))
            X_train_fcwt, y_train_fcwt, X_test_fcwt, y_test_fcwt, X_val_fcwt, y_val_fcwt, le = get_split(fcwt_images,
                                                                                                         labels,
                                                                                                         cross_val_nr,
                                                                                                         emo)
            X_train_stft, y_train_stft, X_test_stft, y_test_stft, X_val_stft, y_val_stft, le = get_split(stft_images,
                                                                                                         labels,
                                                                                                         cross_val_nr,
                                                                                                         emo)
            cv_model, cv_one_acc = start_training_pipeline(X_train_fcwt, y_train_fcwt, X_test_fcwt, y_test_fcwt,
                                                           X_val_fcwt, y_val_fcwt,
                                                           X_train_stft, y_train_stft, X_test_stft, y_test_stft,
                                                           X_val_stft, y_val_stft,
                                                           le, sigma, [], 0, 0, fs, f0, f1, fn, emo)
            print("Accuracy of this cv: %s" % cv_one_acc)
            cv_all_acc.append(cv_one_acc)
            print(cv_all_acc)
            print("Current best CV acc: %s" % cv_best_acc)
            if cv_one_acc > cv_best_acc:
                print("Current cv acc: %s is better than best cv acc: %s" % (cv_one_acc, cv_best_acc))
                best_cv_model.load_state_dict(cv_model.state_dict())
                cv_best_acc = cv_one_acc
                print("New best CV acc: %s " % (cv_best_acc))
        average_acc_sigma = sum(cv_all_acc) / len(cv_all_acc)
        print("Done with sigma %s, average accuracy is: %s" % (sigma, average_acc_sigma))
        sigma_accuracies.append(average_acc_sigma)
        print(sigma_accuracies)
        print("Current best sigma accuracy: %s" % best_acc_sigma)
        print("Current sigma accuracy is: %s" % average_acc_sigma)
        if average_acc_sigma > best_acc_sigma:
            print("Current sigma acc: %s is better than best sigma acc: %s" % (average_acc_sigma, best_acc_sigma))
            best_sigma_model.load_state_dict(best_cv_model.state_dict())
            best_sigma = sigma
            best_acc_sigma = average_acc_sigma
    if emo:
        np.save(arr=sigma_accuracies, file='EMODB_MODELS\sigma_range_test\sigma_accuries')
        if save:
            save_path_best_sigma_model = 'EMODB_MODELS\sigma_range_test\sigma_best_model_sig%s_acc%s' % (
            best_sigma, best_acc_sigma)
            torch.save(best_sigma_model.state_dict(), save_path_best_sigma_model)
    else:
        np.save(arr=sigma_accuracies, file='ENTERFACE_MODELS\sigma_range_test\sigma_accuries')
        if save:
            save_path_best_sigma_model = 'ENTERFACE_MODELS\sigma_range_test\sigma_best_model_sig%s_acc%s' % (
                best_sigma, best_acc_sigma)
            torch.save(best_sigma_model.state_dict(), save_path_best_sigma_model)
    return best_sigma_model, best_sigma


def cv_da_models(emo, rcs, sigma, gn_db, gni, multiple_sigma):
    if emo:
        classes = 7
        nr_of_groups = 10
        fs = 16000
        f1 = 7982
    else:
        classes = 6
        nr_of_groups = 9
        fs = 44100
        f1 = 9785
    cv_best_acc_fcwt = 0
    cv_all_acc_fcwt = []
    cv_all_cm_fcwt = []
    best_cv_model_fcwt = create_model('alexnet', classes, pretrained=True)
    cv_best_acc_stft = 0
    cv_all_acc_stft = []
    cv_all_cm_stft = []
    best_cv_model_stft = create_model('alexnet', classes, pretrained=True)
    for cross_val_nr in range(nr_of_groups):
        print("Testing sigma %s, crossval %s/10" % (sigma, cross_val_nr + 1))
        X_train, y_train, X_test, y_test, X_val, y_val, le = get_split(cross_val_nr, emo)
        fcwt_train_images, stft_train_images, train_labels = create_images.generate_fcwt_stft_images(X_train, y_train,
                                                                                                     out_size=227,
                                                                                                     fs=fs, f0=1, f1=f1,
                                                                                                     fn=227,
                                                                                                     mor_sig=sigma,
                                                                                                     gn_db=gn_db,
                                                                                                     rcs=rcs, gni=0,
                                                                                                     emo=emo,
                                                                                                     multiple_sigma=multiple_sigma,
                                                                                                     save=False,
                                                                                                     to_int=False)
        fcwt_train_images, stft_test_images, test_labels = create_images.generate_fcwt_stft_images(X_test, y_test,
                                                                                                  out_size=227,
                                                                                                  fs=fs, f0=1, f1=f1,
                                                                                                  fn=227,
                                                                                                  mor_sig=sigma,
                                                                                                  gn_db=[], rcs=0,
                                                                                                  gni=0,
                                                                                                  emo=emo,
                                                                                                  multiple_sigma=[],
                                                                                                  save=False,
                                                                                                  to_int=False)
        fcwt_train_images, stft_val_images, val_labels = create_images.generate_fcwt_stft_images(X_val, y_val,
                                                                                               out_size=227,
                                                                                               fs=fs, f0=1, f1=f1,
                                                                                               fn=227,
                                                                                               mor_sig=sigma, gn_db=[],
                                                                                               rcs=0, gni=0,
                                                                                               emo=emo,
                                                                                               multiple_sigma=[],
                                                                                               save=False, to_int=False)

        cv_model_fcwt, cv_one_acc_fcwt, cv_one_cm_fcwt, cv_model_stft, cv_one_acc_stft, cv_one_cm_stft = start_training_pipeline(
            stft_train_images, train_labels, stft_test_images, test_labels, stft_val_images, val_labels,
            le, sigma, [], 5, 0, 44100, 1, 9785, 227, emo)
        print("Accuracy of this fcwt cv: %s" % cv_one_acc_fcwt)
        cv_all_acc_fcwt.append(cv_one_acc_fcwt)
        cv_all_cm_fcwt.append(cv_one_cm_fcwt)
        print(cv_all_cm_fcwt)
        print(cv_all_acc_fcwt)
        print("Current best CV fcwt acc: %s" % cv_best_acc_fcwt)
        if cv_one_acc_fcwt > cv_best_acc_fcwt:
            print("Current fcwt cv acc: %s is better than best cv acc: %s" % (cv_one_acc_fcwt, cv_best_acc_fcwt))
            best_cv_model_fcwt.load_state_dict(cv_model_fcwt.state_dict())
            cv_best_acc_fcwt = cv_one_acc_fcwt
            print("New best fcwt CV acc: %s " % (cv_best_acc_fcwt))

        print("Accuracy of this stft cv: %s" % cv_one_acc_stft)
        cv_all_acc_stft.append(cv_one_acc_stft)
        cv_all_cm_stft.append(cv_one_cm_stft)
        print(cv_all_acc_stft)
        print(cv_all_cm_stft)
        print("Current best CV stft acc: %s" % cv_best_acc_stft)
        if cv_one_acc_stft > cv_best_acc_stft:
            print("Current stft cv acc: %s is better than best cv acc: %s" % (cv_one_acc_stft, cv_best_acc_stft))
            best_cv_model_stft.load_state_dict(cv_model_stft.state_dict())
            cv_best_acc_stft = cv_one_acc_stft
            print("New best stft CV acc: %s " % (cv_best_acc_stft))

    average_acc_sigma_fcwt = sum(cv_all_acc_fcwt) / len(cv_all_acc_fcwt)
    average_acc_sigma_stft = sum(cv_all_acc_stft) / len(cv_all_acc_stft)

    if emo:
        np.save(arr=cv_all_acc_fcwt, file="EMODB_MODELS\EMO_FCWT_CV_ACCS_RCS%s_GNDB%s" % (rcs, gni))
        np.save(arr=cv_all_cm_fcwt, file="EMODB_MODELS\EMO_FCWT_CV_CMS_RCS%s_GNDB%s" % (rcs, gni))
        torch.save(best_cv_model_fcwt.state_dict(),
                  "EMODB_MODELS\EMO_FCWT_MODEL_RCS%s_GNDB%s_acc%s" % (rcs, gni, average_acc_sigma_fcwt))

        np.save(arr=cv_all_acc_stft, file="EMODB_MODELS\EMO_STFT_CV_ACCS_RCS%s_GNDB%s" % (rcs, gni))
        np.save(arr=cv_all_cm_stft, file="EMODB_MODELS\EMO_STFT_CV_CMS_RCS%s_GNDB%s" % (rcs, gni))
        torch.save(best_cv_model_stft.state_dict(),
                  "EMODB_MODELS\EMO_stft_MODEL_RCS%s_GNDB%s_acc%s" % (rcs, gni, average_acc_sigma_stft))
    else:
        np.save(arr=cv_all_acc_fcwt, file="ENTERFACE_MODELS\ENT_FCWT_CV_ACCS_RCS%s_GNDB%s" % (rcs, gni))
        np.save(arr=cv_all_cm_fcwt, file="ENTERFACE_MODELS\ENT_FCWT_CV_CMS_RCS%s_GNDB%s" % (rcs, gni))
        torch.save(best_cv_model_fcwt.state_dict(),
                  "ENTERFACE_MODELS\ENT_FCWT_MODEL_RCS%s_GNDB%s_acc%s" % (rcs, gni, average_acc_sigma_fcwt))

        np.save(arr=cv_all_acc_stft, file="ENTERFACE_MODELS\ENT_STFT_CV_ACCS_RCS%s_GNDB%s" % (rcs, gni))
        np.save(arr=cv_all_cm_stft, file="ENTERFACE_MODELS\ENT_STFT_CV_CMS_RCS%s_GNDB%s" % (rcs, gni))
        torch.save(best_cv_model_stft.state_dict(),
                  "ENTERFACE_MODELS\ENT_stft_MODEL_RCS%s_GNDB%s_acc%s" % (rcs, gni, average_acc_sigma_stft))



cv_da_models(True, rcs=5, sigma=0, gn_db=[], gni='NO_STFT_BASELINE', multiple_sigma=[])