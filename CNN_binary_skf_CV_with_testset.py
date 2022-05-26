# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:12:15 2021

@author: gvasam
"""


import numpy as np 
import pandas as pd 
import seaborn as sns
import math

from sklearn.model_selection import *
import os
import matplotlib.pyplot as plt
from sklearn.metrics import *


from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
import tensorflow as tf



from tensorflow.keras.applications import *



def prepare_df(root_dir):
    image_df = pd.DataFrame()
    image_df['filename'] = ''
    image_df['class_names'] = None
    image_df['patient_ID'] = None
    # Populate absoulte paths of filenames and class labels in a dataframe called 'image_df'
    for dirname, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path_name = os.path.join(dirname, filename)
            # Class name is the name of the directory: like MVM+, MVM-
            class_label = dirname.split('/')[-1]
            patient_ID = filename.split('_')[0]
            image_df = image_df.append({'filename':path_name, 'class_names':class_label,'patient_ID':patient_ID},ignore_index=True)
            
    return image_df


root_dir = '/DATA/MVM' #edit this


image_df = prepare_df(root_dir)


image_df.class_names.unique()


def create_model(img_size=256,channels=3,classes=3):
    model = Sequential()
    
    input_shape = (img_size,img_size,channels)
    base_model = InceptionV3(input_shape=input_shape,classes=3,include_top=False,pooling='avg', classifier_activation=None)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    
    model.compile(optimizers.Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
   
    
    return model


img_width = img_height = 512
batch_size = 2
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
fold_ = 0
histories = []
predictions = []
ground_truth = []
pred_prob = []

# patient_IDs = image_df['patient_ID'].drop_duplicates()

#If you have a dedicated test set, comment below line and read the test set
train_val_data, testData = train_test_split(image_df, test_size=0.2,random_state=42)
train_patient_class_dict = pd.Series(train_val_data.class_names.values,index=train_val_data.patient_ID).to_dict()
train_patient_IDs = train_val_data['patient_ID'].drop_duplicates()
#test_df = prepare_df(test_root_dir)

for train_index, _ in skf.split(list(train_patient_class_dict.keys()), list(train_patient_class_dict.values())):
    #Split the dataframe into train and test
    train_patients = [train_patient_IDs.values[i] for i in train_index]
    trainData = train_val_data[train_val_data['patient_ID'].isin(train_patients)]
    valData = train_val_data[~train_val_data['patient_ID'].isin(train_patients)]
    print('Initializing Kfold %s'%str(fold_))
    print('Train shape:',trainData.shape)
    print('Val shape:',valData.shape)
    print('Test shape:',testData.shape)
    epochs = 100
    # Get the train generator  with a validation split of 0.2 along with all augmentations
    train_datagen = ImageDataGenerator(rescale=1./255,zoom_range = 0.2,
                                   rotation_range = 30,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   brightness_range = [0.8, 1.2],
                                   channel_shift_range = 150.0,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = 'reflect')    
    
    val_datagen = ImageDataGenerator(rescale=1./255,zoom_range = 0.2,
                                   rotation_range = 30,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   brightness_range = [0.8, 1.2],
                                   channel_shift_range = 150.0,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = 'reflect')
    
    #Get the test generator
    test_datagen = ImageDataGenerator(rescale=1. / 255) 
    
    # Data is accessed using the dataframe. Use the subset='training' for train_generator 
    train_generator=train_datagen.flow_from_dataframe(
    dataframe=trainData,
    directory=None,
    x_col="filename",
    y_col="class_names",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    target_size=(img_width, img_height))

    # Data is accessed using the dataframe. Use the subset='validation' for validation_generator
    validation_generator=train_datagen.flow_from_dataframe(
        dataframe=valData,
        directory=None,
        x_col="filename",
        y_col="class_names",
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
        target_size=(img_width, img_height))

    #Get test generator
    test_generator=test_datagen.flow_from_dataframe(
        dataframe=testData,
        directory=None,
        x_col="filename",
        y_col="class_names",
        batch_size=1,
        shuffle=False,
        class_mode="categorical",
        target_size=(img_width, img_height))
    
    # Create a model with two classes
    
    model = create_model(img_size=512,classes=2)
    
    STEP_SIZE_TRAIN=math.ceil(train_generator.n//train_generator.batch_size)
    STEP_SIZE_VALID=math.ceil(validation_generator.n//validation_generator.batch_size)
    STEP_SIZE_TEST=math.ceil(test_generator.n//test_generator.batch_size)
    
    #Model checkpoint to save model for each fold
    checkpoint_filepath = 'model_fold_{0}'.format(fold_)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=validation_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=epochs, callbacks=[model_checkpoint_callback]
    )
    # Append history of each model
    histories.append(history)
    #Predict probabitlies of each class for the test set
    pred = model.predict_generator(test_generator,steps = STEP_SIZE_TEST,verbose=1)
    pred_prob.append(np.max(pred,axis=1))
    #Get the predicted class
    predicted_class_indices=np.argmax(pred,axis=1)
    # Get the groun truth
    labels=test_generator.classes
    
    # Save predictions and groundtruth in thier respective lists
    predictions.append(predicted_class_indices)
    ground_truth.append(labels)
    fold_+=1


for fold in range(5):
    history = histories[fold]
    f,ax = plt.subplots(1,2, figsize = (15,5))
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'test'], loc='upper left')
    
    # summarize history for loss
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'test'], loc='upper left')
    f.suptitle('History for fold {0}'.format(fold))
    plt.savefig('History for fold {0}.tiff'.format(fold), format="tiff", dpi=600)
    plt.show()
    plt.close('all')


labels = list(train_generator.class_indices.keys())


labels


import statistics

# Get final predicitons by taking mean or mode from each model
predictions_ = np.array(predictions)
final_predictions = []
for i in range(predictions_.shape[1]):
    try:
        final_predictions.append(statistics.mean(predictions_[:,i]))
    except:
        final_predictions.append(np.random.choice(predictions_[:,i],1)[0])

# final_predictions = [statistics.mode(predictions_[:,i]) for i in range(predictions_.shape[1])]

final_acc = accuracy_score(ground_truth[0], final_predictions)

final_acc


# Confusion matrix and average CM
all_cms = []

for i in range(5):
    title = 'Confusion Matrix for fold {0}'.format(i)
    cm = confusion_matrix(ground_truth[i],predictions[i])
    all_cms.append(cm)
    df_cm = pd.DataFrame(cm, index = labels,
                  columns = labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, robust=True)
    plt.title(title)
    plt.savefig('Confusion Matrix for fold {0}.tiff'.format(fold), format="tiff", dpi=600)
    plt.show()
    plt.close('all')


avg_cm = np.mean(all_cms, axis=0)

avg_cm

df_cm = pd.DataFrame(avg_cm, index = labels,
                  columns = labels)
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True,  robust=True)
plt.title('Average Confusion matrix across all folds')
plt.savefig('Average Confusion matrix across all folds.tiff', format="tiff", dpi=600)
plt.show()
plt.close('all')



# Classification output on each image

from PIL import *

def predict_class(img_path):

  img_pil = Image.open(img_path)
  img_pil = img_pil.resize((512,512))

  img_tensor = tf.keras.preprocessing.image.img_to_array(img_pil)
  img_tensor = img_tensor*(1/255.0)
  img_tensor = tf.expand_dims(img_tensor,axis=0)

  pred = model.predict(img_tensor)
  pred = np.argmax(pred,axis=1)

  return pred[0]


labels


def get_test_predictions(testData):

  test_pred_df = testData.copy()
  test_pred_df['Predicted class'] = None

  for index in test_pred_df.index:
    
    img_path = test_pred_df.loc[index,'filename']
    pred = predict_class(img_path)
    
    test_pred_df.loc[index,'Predicted class'] = labels[pred]

  return test_pred_df


test_pred_df = get_test_predictions(testData)

test_pred_df


# To assess the robutness of the model
# Estimate mean accuracy and stdev
accs = []
for i in range(5):
    accs.append(accuracy_score(ground_truth[i],predictions[i]))
accs = np.array(accs)
print('Mean Accuracy : {0}, Std : {1}'.format(accs.mean(), accs.std()))



# Accuracy, precision, recall and F1 Score

def get_metrics(gt,pred):
  metrics_df = pd.DataFrame(columns=['Type','Accuracy','Precision','Recall','F1 Score'])
  types = ['weighted','macro','micro']
  accuracy = accuracy_score(gt,pred)
  for type_ in types:
    precision, recall, fscore, _ = precision_recall_fscore_support(gt,pred,average=type_)
    metrics_df = metrics_df.append({'Type':type_,'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1 Score':fscore},ignore_index=True)
  return metrics_df


for i in range(5):
  metrics_df = get_metrics(ground_truth[i], predictions[i])
  print('****************** Fold {0} ******************'.format(i))
  print(metrics_df.to_markdown())



# Precision-Recall and ROC curves

from tensorflow.keras.utils import to_categorical
from itertools import cycle


# ROC curves
def get_roc_curve(labels, predicted_vals, ground_truth):
    auc_roc_vals = []
    predicted_vals = np.array(predicted_vals)
    ground_truth = np.array(ground_truth)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(labels)
    for i in range(len(labels)):
#         try:
        gt = ground_truth[:, i]
        pred = predicted_vals[:, i]
        roc_auc[i] = roc_auc_score(gt, pred)
        
        fpr[i], tpr[i], _ = roc_curve(gt, pred)
         
    fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth.ravel(), predicted_vals.ravel())
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average them and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr


    # Plot all ROC curves
    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for classes: "+ ', '.join(labels))
    plt.legend(loc="lower right")
    plt.savefig('ROC Curve.tiff', format='tiff', dpi=600)
    plt.show()
    plt.close('all')


# Precision-Recall curves

def get_pr_curve(labels, predicted_vals, ground_truth):
    auc_roc_vals = []
    predicted_vals = np.array(predicted_vals)
    ground_truth = np.array(ground_truth)
    lr_precision = dict()
    lr_recall = dict()
    
    n_classes = len(labels)
    for i in range(len(labels)):
#         try:
        gt = ground_truth[:, i]
        pred = predicted_vals[:, i]
               
        lr_precision[i], lr_recall[i], _ = precision_recall_curve(gt, pred)
         
    lr_precision["micro"], lr_recall["micro"], _ = precision_recall_curve(ground_truth.ravel(), predicted_vals.ravel())
    # First aggregate all false positive rates
    all_lr_precision = np.unique(np.concatenate([lr_precision[i] for i in range(n_classes)]))

    # Then interpolate all PR curves at these points
    mean_lr_recall = np.zeros_like(all_lr_precision)
    for i in range(n_classes):
        mean_lr_recall += np.interp(all_lr_precision, lr_precision[i], lr_recall[i])

    # Finally average them and compute AUC
    mean_lr_recall /= n_classes

    lr_precision["macro"] = all_lr_precision
    lr_recall["macro"] = mean_lr_recall


    # Plot all PR curves
    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(
        lr_recall["micro"],
        lr_precision["micro"],        
        label="micro-average PR curve",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        lr_recall["macro"],
        lr_precision["macro"],
        label="macro-average PR curve",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            lr_recall[i],
            lr_precision[i],            
            color=color,
            lw=lw,
            label="PR curve of class {0}".format(labels[i]),
        )

    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision Recall Curve for classes: "+ ', '.join(labels))
    plt.legend(loc="lower right")
    plt.savefig('PR Curve.tiff', format='tiff', dpi=600)
    plt.show()
    plt.close('all')




test_generator=test_datagen.flow_from_dataframe(
        dataframe=testData,
        directory=None,
        x_col="filename",
        y_col="class_names",
        batch_size=1,
        shuffle=False,
        class_mode="categorical",
        target_size=(img_width, img_height))
    
    # Create a model with two classes
    
model = create_model(img_size=512,classes=2)


ground_truth_test = test_generator.classes


ground_truth_test = to_categorical(ground_truth_test)


all_preds = []
for i in range(5):
  model.load_weights('model_fold_{0}'.format(i))
  preds = model.predict(test_generator)
  all_preds.append(preds)
  print('****************** Fold {0} ******************'.format(i))
  get_roc_curve(labels, preds, ground_truth_test)
  get_pr_curve(labels, preds, ground_truth_test)


# Mean ROC and PR curves for all folds

all_preds_mean = np.mean(all_preds, axis=0)
all_preds_mean.shape

get_roc_curve(labels, all_preds_mean, ground_truth_test)

get_pr_curve(labels, all_preds_mean, ground_truth_test)

"""
#plot overall (average) confusion matrix
t = 'Confusion Matrix'
conf_mat = confusion_matrix(ground_truth[0], final_predictions)
df_conf_mat = pd.DataFrame(conf_mat, index = labels, columns = labels)
plt.figure(figsize = (10,7))
sns.heatmap(df_conf_mat, annot=True)
plt.title(t)
plt.savefig('Confusion Matrix')
plt.show()
plt.close('all')


#plot confusion matrix for each fold
for i in range(5):
    title = 'Confusion Matrix for fold {0}'.format(i)
    cm = confusion_matrix(ground_truth[i],predictions[i])
    df_cm = pd.DataFrame(cm, index = labels,
                  columns = labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.title(title)
    plt.savefig('Confusion Matrix for fold {0}.tiff'.format(i), format='tiff', dpi=600)
    plt.show()
    plt.close('all')

# Estimate mean accuracy and stdev
accs = []
for i in range(5):
    accs.append(accuracy_score(ground_truth[i],predictions[i]))
accs = np.array(accs)
print('Mean Accuracy : {0}, Std : {1}'.format(accs.mean(), accs.std()))

# Estimate mean precision and stdev
prec = []
for i in range(5):
    prec.append(precision_score(ground_truth[i],predictions[i]))
prec = np.array(prec)
print('Mean Precision : {0}, Std : {1}'.format(prec.mean(), prec.std()))

# Estimate mean recall and stdev
recall = []
for i in range(5):
    recall.append(recall_score(ground_truth[i],predictions[i]))
recall = np.array(recall)
print('Mean Recall Score : {0}, Std : {1}'.format(recall.mean(), recall.std()))

# Estimate mean f1_score and stdev
fone = []
for i in range(5):
    fone.append(f1_score(ground_truth[i],predictions[i]))
fone = np.array(fone)
print('Mean f1_score : {0}, Std : {1}'.format(fone.mean(), fone.std()))

"""
