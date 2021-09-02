import numpy as np
from sklearn import metrics
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from MyDataGenerators import DataGeneratorVariableLength as DataGenerator

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

SMALL_ERRORS = True

if SMALL_ERRORS:
    StarIndices = np.load('../data/small_error_indicies_random.npy')
else:
    StarIndices = np.load('../data/StarIndexList.npy')
HaloLabels = np.load('../data/StarLabels_accreted_virialized.npy')
ltrue = np.load('../data/sliced_l_true.npy')
btrue = np.load('../data/sliced_b_true.npy')
pmra = np.load('../data/sliced_pmra.npy')
pmdec = np.load('../data/sliced_pmdec.npy')
pmra_error = np.load('../data/sliced_pmra_error.npy')
pmdec_error = np.load('../data/sliced_pmdec_error.npy')
parallax = np.load('../data/sliced_parallax.npy')
parallax_error = np.load('../data/sliced_parallax_error.npy')
feh = np.load('../data/sliced_feh.npy')

radial_velocity = np.load('../data/sliced_radial_velocity.npy')
StarIndices = StarIndices[np.isfinite(radial_velocity[StarIndices])]

NumHalo = np.sum(HaloLabels[StarIndices])
TotalStars = len(StarIndices)
NumNonHalo = TotalStars - NumHalo

BATCHSIZE = int(TotalStars / NumHalo * 5)

l_mean, l_std = 0, 0
b_mean, b_std = 0, 0
pmra_mean, pmra_std = 0, 0
pmdec_mean, pmdec_std = 0, 0
normstatistics = [[], [], [], [], [], []]
with open('../data/MeansAndVariance.txt') as f:
    for line in f:
        line = line.strip().split(',')

        if line[0] == 'l_true':
            l_mean, l_std = float(line[1]), np.sqrt(float(line[2]))
            normstatistics[0] = [l_mean, l_std]
        if line[0] == 'b_true':
            b_mean, b_std = float(line[1]), np.sqrt(float(line[2]))
            normstatistics[1] = [b_mean, b_std]
        if line[0] == 'parallax':
            parallax_mean, parallax_std = float(line[1]), np.sqrt(float(line[2]))
            normstatistics[2] = [parallax_mean, parallax_std]
        if line[0] == 'pmra_true':
            pmra_mean, pmra_std = float(line[1]), np.sqrt(float(line[2]))
            normstatistics[3] = [pmra_mean, pmra_std]
        if line[0] == 'pmdec_true':
            pmdec_mean, pmdec_std = float(line[1]), np.sqrt(float(line[2]))
            normstatistics[4] = [pmdec_mean, pmdec_std]
        if line[0] == 'feh':
            feh_mean, feh_std = float(line[1]), np.sqrt(float(line[2]))
            normstatistics[5] = [feh_mean, feh_std]

star_weights = {0: float(TotalStars) / NumNonHalo,
                1: float(TotalStars) / NumHalo / 5
                }
print(star_weights)
print('Halo stars: {0}'.format(NumHalo))
print('Non-Halo stars: {0}'.format(NumNonHalo))
print('Batch size of {0}'.format(BATCHSIZE))

TrainingData = DataGenerator(list_IDs=StarIndices[:-2000000],
                             data=[ltrue,
                                   btrue,
                                   parallax,
                                   pmra,
                                   pmdec,
                                   feh,
                                   np.zeros([ltrue.shape[0],1]),  # error on l
                                   np.zeros([btrue.shape[0],1]),  # error on b
                                   parallax_error,
                                   pmra_error,  # error on pmra
                                   pmdec_error,  # error on pmdec
                                   np.zeros([feh.shape[0], 1]),  # metallicity
                                   ],
                             normstatistics=normstatistics,
                             labels=HaloLabels,
                             batch_size=BATCHSIZE,
                             class_weights=star_weights,
                             shuffle=True,
                             random_sample_size=20)
print('Examine first set')
print(TrainingData[0])
ValidationData = DataGenerator(list_IDs=StarIndices[-2000000:-1000000],
                               data=[ltrue,
                                     btrue,
                                     parallax,
                                     pmra,
                                     pmdec,
                                     feh,
                                     np.zeros([ltrue.shape[0],1]),  # error on l
                                     np.zeros([btrue.shape[0],1]),  # error on b
                                     parallax_error,
                                     pmra_error,  # error on pmra
                                     pmdec_error,  # error on pmdec
                                     np.zeros([feh.shape[0], 1]),  # metallicity
                                     ],
                               normstatistics=normstatistics,
                               labels=HaloLabels,
                               batch_size=BATCHSIZE,
                               class_weights=star_weights,
                               shuffle=True,
                               random_sample_size=20)

print('Examine validation set')
print(ValidationData[0])
TestData = DataGenerator(list_IDs=StarIndices[-1000000:],
                         data=[ltrue,
                               btrue,
                               parallax,
                               pmra,
                               pmdec,
                               feh,
                               np.zeros([ltrue.shape[0],1]),  # error on l
                               np.zeros([btrue.shape[0],1]),  # error on b
                               parallax_error,
                               pmra_error,  # error on pmra
                               pmdec_error,  # error on pmdec
                               np.zeros([feh.shape[0], 1]),  # metallicity
                               ],
                         normstatistics=normstatistics,
                         labels=HaloLabels,
                         batch_size=BATCHSIZE,
                         class_weights=star_weights,
                         shuffle=True,
                         random_sample_size=1
                         )

REGL = 0.002
EPCOCHS = 400
LR = 0.001
LRDECAY = 0.5 * LR / EPCOCHS / len(TrainingData)

filepath = "saved_models/SmallSet_AccretedVirializedFiveDFeH-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1,
                              patience=5, min_lr=1.0e-6)
es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

callbacks_list = [reduce_lr, es]  # checkpoint,

model = Sequential()
model.add(Dense(100,
                activation='relu',
                kernel_initializer='normal',
                input_dim=6,
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
                              # validation_data=ValidationData,
                              epochs=1,
                              use_multiprocessing=True,
                              workers=8,
                              class_weight=star_weights,
                              # callbacks=callbacks_list,
                              verbose=1
                              )
print('Done with initial training run')
history = model.fit_generator(generator=TrainingData,
                              validation_data=ValidationData,
                              epochs=EPCOCHS,
                              use_multiprocessing=True,
                              workers=8,
                              class_weight=star_weights,
                              callbacks=callbacks_list,
                              verbose=2
                              )

model.save('saved_models/SmallSet_AccretedVirializedFiveDFeH_5.h5')

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
plt.title('Measured Parallax [small errors] Accreted VirializedFiveDFeH')
plt.legend(loc='best', frameon=False, fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('../plots/History_SmallSet_AccretedVirializedFiveDFeH_5.pdf',
            bbox_inches='tight')


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
plt.title('Validation Accreted Virialized FiveDFeH (Measured Parallax) [Small errors]')
plt.xlabel('Network output')
plt.ylabel('Stars per bin')
plt.yscale('log')
plt.savefig('../plots/MP_SmallSet_AccretedVirializedFiveDFeH_5_Validation_NNout.pdf',
            bbox_inches='tight'
            )

plt.figure(figsize=(4,4))
plt.hist(Test_Preds[Test_Labels==0], bins=100, range=(0,1), histtype='step')
plt.hist(Test_Preds[Test_Labels==1], bins=100, range=(0,1), histtype='step')
plt.plot([],[], color='C0', label='Non halo')
plt.plot([],[], color='C1', label='Halo')
plt.legend(loc='best', frameon=False)
plt.title('Test Accreted Virialized FiveDFeH (Measured Parallax) [Small errors]')
plt.xlabel('Network output')
plt.ylabel('Stars per bin')
plt.yscale('log')
plt.savefig('../plots/MP_SmallSet_AccretedVirializedFiveDFeH_5_Test_NNout.pdf',
            bbox_inches='tight'
            )

print('Generating the ROC curve for the test set')
fpr, tpr, thresholds = metrics.roc_curve(Test_Labels, Test_Preds)
TestAUC = metrics.auc(fpr, tpr)

plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, label='AUC={0:.4f}'.format(TestAUC))
plt.title('Test Accreted Virialized FiveDFeH (Measured Parallax) [Small errors]')
plt.xscale('log')
plt.ylim(0,1.2)
plt.legend(loc='upper left', frameon=False, fontsize=12)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('../plots/MP_SmallSet_AccretedVirializedFiveDFeH_5_ROC_Curve.pdf',
            bbox_inches='tight'
            )
