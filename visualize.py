def ROC_curve(data_test, label_test, weights, model):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # use the model to do classifications
    label_predict = model.model.predict(data_test)
    fpr, tpr, _ = roc_curve(
        label_test, label_predict[:, 0], sample_weight=weights)  # calculate the ROC curve
    roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='k', label='random chance')
    plt.plot(tpr, fpr, lw=2, color='cyan', label='NN auc = %.3f' % (roc_auc))
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('true positive rate')
    plt.ylabel('false positive rate')
    plt.title('receiver operating curve')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig('plots/ROC_'+model.name+'.pdf')


def trainingPlots(history, model):
    import matplotlib.pyplot as plt
    # plot loss vs epoch
    ax = plt.subplot(2, 1, 1)
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.legend(loc="upper right")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    # plot accuracy vs epoch
    ax = plt.subplot(2, 1, 2)
    ax.plot(history.history['acc'], label='acc')
    ax.plot(history.history['val_acc'], label='val_acc')
    ax.legend(loc="upper left")
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    plt.savefig('plots/trainingPlot_'+model.name+'.pdf')


def discPlot(model, sig, bkg):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    sig = sig.values[:, 0:model.ninp]
    bkg = bkg.values[:, 0:model.ninp]

    sig = StandardScaler().fit_transform(sig)
    bkg = StandardScaler().fit_transform(bkg)

    sig_pred = model.model.predict(sig)
    bkg_pred = model.model.predict(bkg)

    plt.figure(figsize=(12, 8))
    plt.title('NN Discriminant')
    plt.xlabel('NN Disc.')
    plt.ylabel('Events/Bin')
    plt.hist(bkg_pred, histtype='step', color='red', label='ZTT', bins=100)
    plt.hist(sig_pred, histtype='step', color='blue', label='VBF', bins=100)
    plt.legend()
    plt.savefig('plots/disc_'+model.name+'.pdf')
