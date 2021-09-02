import numpy as np
from sklearn import metrics
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from MyDataGenerators import DataGenerator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Set the plotting styles

plt.rcParams.update({'font.family': 'cmr10', 'font.size': 13})
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 15
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


StarIndices = np.load('../data/StarIndexList.npy',
                      # mmap_mode='r'
                      )
HaloLabels = np.load('../data/StarLabels.npy')
ltrue = np.load('../data/sliced_l_true.npy')
btrue = np.load('../data/sliced_b_true.npy')
parallax = np.load('../data/sliced_parallax_true.npy')
pmratrue = np.load('../data/sliced_pmra_true.npy')
pmdectrue = np.load('../data/sliced_pmdec_true.npy')

l_mean, l_std = 0, 0
b_mean, b_std = 0, 0
parallax_mean, parallax_std = 0, 0
pmra_mean, pmra_std = 0, 0
pmdec_mean, pmdec_std = 0, 0
normstatistics = []
with open('../data/MeansAndVariance.txt') as f:
    for line in f:
        line = line.strip().split(',')

        if line[0] == 'l_true':
            l_mean, l_std = float(line[1]), np.sqrt(float(line[2]))
            # print 'l: {0}, {1}'.format(l_mean, l_std)
            normstatistics.append([l_mean, l_std])
        if line[0] == 'b_true':
            b_mean, b_std = float(line[1]), np.sqrt(float(line[2]))
            # print 'b: {0}, {1}'.format(b_mean, b_std)
            normstatistics.append([b_mean, b_std])
        if line[0] == 'parallax_true':
            parallax_mean = float(line[1])
            parallax_std = np.sqrt(float(line[2]))
            # print 'parallax_true: {0}, {1}'.format(parallax_mean, parallax_std)
            normstatistics.append([parallax_mean, parallax_std])
        if line[0] == 'pmra_true':
            pmra_mean, pmra_std = float(line[1]), np.sqrt(float(line[2]))
            # print 'pmra_true: {0}, {1}'.format(pmra_mean, pmra_std)
            normstatistics.append([pmra_mean, pmra_std])
        if line[0] == 'pmdec_true':
            pmdec_mean, pmdec_std = float(line[1]), np.sqrt(float(line[2]))
            # print 'pmdec_true: {0}, {1}'.format(pmdec_mean, pmdec_std)
            normstatistics.append([pmdec_mean, pmdec_std])


star_weights = {0: 598334698.0 / (598334698.0-700280),
                1: 598334698.0 / 700280
                }

TrainingData = DataGenerator(list_IDs=StarIndices[:-2000000],
                             data=[ltrue, btrue, parallax,
                                   pmratrue, pmdectrue],
                             normstatistics=normstatistics,
                             labels=HaloLabels,
                             batch_size=65536,
                             class_weights=star_weights,
                             shuffle=True)
ValidationData = DataGenerator(list_IDs=StarIndices[-2000000:-1000000],
                               data=[ltrue, btrue, parallax,
                                     pmratrue, pmdectrue],
                               normstatistics=normstatistics,
                               labels=HaloLabels,
                               batch_size=65536,
                               class_weights=star_weights,
                               shuffle=True)
TestData = DataGenerator(list_IDs=StarIndices[-2000000:-1000000],
                         data=[ltrue, btrue, parallax,
                               pmratrue, pmdectrue],
                         normstatistics=normstatistics,
                         labels=HaloLabels,
                         batch_size=65536,
                         class_weights=star_weights,
                         shuffle=True)

REGL = 0.002
EPCOCHS = 400
LR = 0.001
LRDECAY = 0.5 * LR / EPCOCHS / len(TrainingData)

filepath = "saved_models/UsingParallaxTrue_NoRegLRPlateau_LargeBatch_ES-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1,
                              patience=5, min_lr=1.0e-6)
es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

callbacks_list = [checkpoint, reduce_lr, es]

model = Sequential()
model.add(Dense(100,
                activation='relu',
                kernel_initializer='normal',
                input_dim=5,
                # kernel_regularizer=regularizers.l2(REGL)
                )
          )
model.add(Dense(100,
                activation='relu',
                # kernel_regularizer=regularizers.l2(REGL)
                )
          )
model.add(Dense(100,
                activation='relu',
                # kernel_regularizer=regularizers.l2(REGL)
                )
          )
model.add(Dense(1, activation='sigmoid'))
adam = Adam(lr=LR)
model.compile(optimizer=adam, loss='binary_crossentropy')

history = model.fit_generator(generator=TrainingData,
                              validation_data=ValidationData,
                              epochs=EPCOCHS,
                              use_multiprocessing=True,
                              workers=8,
                              class_weight=star_weights,
                              callbacks=callbacks_list,
                              verbose=1
                              )

model.save('saved_models/UsingParallaxTrue_NoRegLRPlateau_LargeBatch_ES.h5')

with open('testing.txt', 'w') as f:
    f.write('epoch,loss,val_loss\n')
    for i, (x, y) in enumerate(zip(history.history['loss'],
                                   history.history['val_loss']
                                   )):
        f.write('{0},{1},{2}\n'.format(i, x, y))

# Plot the initial training results
plt.figure()
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.title('True parallax')
plt.legend(loc='best', frameon=False, fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('../plots/History_UsingParallaxTrue_NoRegLRPlateau_LargeBatch_ES.pdf', bbox_inches='tight')


Valid_Preds = np.array([])
Valid_Labels = np.array([])
for i in range(len(ValidationData)):
    x, y, weights = ValidationData[i]
    print('Working on batch {0} of {1} for validation predictions'.format(i, len(ValidationData)))
    Valid_Preds = np.append(Valid_Preds, model.predict(x))
    Valid_Labels = np.append(Valid_Labels, np.array(y))

Test_Preds = np.array([])
Test_Labels = np.array([])
for i in range(len(TestData)):
    x, y, weights = TestData[i]
    print('Working on batch {0} of {1}'.format(i, len(TestData)))
    Test_Preds = np.append(Test_Preds, model.predict(x))
    Test_Labels = np.append(Test_Labels, np.array(y))

plt.figure(figsize=(4,4))
plt.hist(Valid_Preds[Valid_Labels==0], bins=100, range=(0,1), histtype='step')
plt.hist(Valid_Preds[Valid_Labels==1], bins=100, range=(0,1), histtype='step')
plt.plot([],[], color='C0', label='Non halo')
plt.plot([],[], color='C1', label='Halo')
plt.legend(loc='best', frameon=False)
plt.title('Validation data (True parallax)')
plt.xlabel('Network output')
plt.ylabel('Stars per bin')
plt.yscale('log')
plt.savefig('../plots/TrueParallax_NoRegLRPlateau_LargeBatch_ES_Validation_NNout.pdf',
            bbox_inches='tight'
            )

plt.figure(figsize=(4,4))
plt.hist(Test_Preds[Test_Labels==0], bins=100, range=(0,1), histtype='step')
plt.hist(Test_Preds[Test_Labels==1], bins=100, range=(0,1), histtype='step')
plt.plot([],[], color='C0', label='Non halo')
plt.plot([],[], color='C1', label='Halo')
plt.legend(loc='best', frameon=False)
plt.title('Test data (True Parallax)')
plt.xlabel('Network output')
plt.ylabel('Stars per bin')
plt.yscale('log')
plt.savefig('../plots/TrueParallaxLRPlateau_LargeBatch_ES_NoReg_Test_NNout.pdf',
            bbox_inches='tight'
            )

print('Generating the ROC curve for the test set')
fpr, tpr, thresholds = metrics.roc_curve(Test_Labels, Test_Preds)
TestAUC = metrics.auc(fpr, tpr)

plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, label='AUC={0:.4f}'.format(TestAUC))
plt.title('Test data (True Parallax)')
plt.xscale('log')
plt.ylim(0,1.2)
plt.legend(loc='upper left', frameon=False, fontsize=12)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('../plots/TrueParallaxLRPlateau_LargeBatch_ES_NoReg_ROC_Curve.pdf',
            bbox_inches='tight'
            )
