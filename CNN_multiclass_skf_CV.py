# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:12:15 2021

@author: gvasam
"""


import numpy as np 
import pandas as pd 
import seaborn as sns

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
from PIL import *
from sklearn.metrics import *


from tensorflow.keras.applications import *
from tensorflow.keras.utils import to_categorical
from itertools import cycle


def prepare_df(root_dir):
    image_df = pd.DataFrame()
    image_df['filename'] = ''
    image_df['class_names'] = None
    image_df['patient_ID'] = None
    # Populate absoulte paths of filenames and class labels in a dataframe called 'image_df'
    for dirname, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path_name = os.path.join(dirname, filename)
            # Class name is the name of the directory: like CLass1, Class2, Class3
            class_label = dirname.split('/')[-1]
            patient_ID = filename.split('_')[0]
            image_df = image_df.append({'filename':path_name, 'class_names':class_label,'patient_ID':patient_ID},ignore_index=True)
            
    return image_df


root_dir = '/DATA/PE_onset' #edit this


image_df = prepare_df(root_dir)


image_df.class_names.unique()

image_df


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


def plot_histories(histories):
  for fold in range(len(histories)):
    history = histories[fold]
    f,ax = plt.subplots(1,2, figsize = (15,5))
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['training', 'validation'], loc='upper left')
    
    # summarize history for loss
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['training', 'validation'], loc='upper left')
    f.suptitle('History for fold {0}'.format(fold))
    plt.savefig('History for fold {0}.tiff'.format(fold), format="tiff", dpi=600)
    plt.show()
    plt.close('all')
    
    
def plot_cm(ground_truth_fold, predictions_fold, fold, labels):  
  title = 'Confusion matrix for fold {0}'.format(fold)
  cm = confusion_matrix(ground_truth_fold, predictions_fold)
  # all_cms.append(cm)
  df_cm = pd.DataFrame(cm, index = labels,
                columns = labels)
  plt.figure(figsize = (10,7))
  sns.heatmap(df_cm, annot=True, robust=True)
  plt.title(title)
  plt.savefig('Confusion matrix for fold {0}.tiff'.format(fold), format="tiff", dpi=600)
  plt.show()
  plt.close('all')
  return cm
    
    
def get_average_cm(all_cms, labels):
  avg_cm = np.mean(all_cms,axis=0)
  df_cm = pd.DataFrame(avg_cm, index = labels,
                  columns = labels)
  plt.figure(figsize = (10,7))
  sns.heatmap(df_cm, annot=True,  robust=True)
  plt.title('Average confusion matrix across all folds')
  plt.savefig('Average confusion matrix across all folds.tiff', format="tiff", dpi=600)
  plt.show()
  plt.close('all')
  

def get_avg_acc_std(ground_truth,predictions):
  # To assess the robutness of the model
  accs = []
  for i in range(len(ground_truth)):
      accs.append(accuracy_score(ground_truth[i],predictions[i]))
  accs = np.array(accs)
  print('Mean : {0}, Std : {1}'.format(accs.mean(), accs.std()))
  
  
def predict_class(img_path):

  img_pil = Image.open(img_path)
  img_pil = img_pil.resize((512,512))

  img_tensor  = tf.keras.preprocessing.image.img_to_array(img_pil)
  img_tensor = img_tensor*(1/255.0)
  img_tensor = tf.expand_dims(img_tensor,axis=0)

  pred = model.predict(img_tensor)
  pred = np.argmax(pred,axis=1)

  return pred[0]

    
def get_test_predictions(testData):

  test_pred_df = testData.copy()
  test_pred_df['Predicted class'] = None

  for index in test_pred_df.index:
    
    img_path = test_pred_df.loc[index,'filename']
    pred = predict_class(img_path)
    
    test_pred_df.loc[index,'Predicted class'] = labels[pred]

  return test_pred_df


def get_metrics(gt,pred):
  metrics_df = pd.DataFrame(columns=['Type','Accuracy','Precision','Recall','F1 Score'])
  types = ['weighted','macro','micro']
  accuracy = accuracy_score(gt,pred)
  for type_ in types:
    precision, recall, fscore, _ = precision_recall_fscore_support(gt,pred,average=type_)
    metrics_df = metrics_df.append({'Type':type_,'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1 Score':fscore},ignore_index=True)
  return metrics_df


def get_roc_curve(labels, predicted_vals, ground_truth, fold):
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

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
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
        label="macro-average ROC curve ",
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
            label="ROC curve of class {0} (area = {1:0.2f})".format(labels[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label="No learning")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fold {0} | ROC curve for classes: "+ ', '.join(labels).format(fold))
    plt.legend(loc="lower right")
    plt.savefig('ROC_fold_{0}.tiff'.format(fold), format = "tiff", dpi=600)
    plt.show()
    plt.close('all')
    
    
def get_pr_curve(labels, predicted_vals, ground_truth, fold):
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

    # Then interpolate all PR curves at this point
    mean_lr_recall = np.zeros_like(all_lr_precision)
    for i in range(n_classes):
        mean_lr_recall += np.interp(all_lr_precision, lr_precision[i], lr_recall[i])

    # Finally average it and compute AUC
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
        label="macro-average PR curve ",
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

    plt.plot([0, 1], [0, 0], "k--", lw=lw, label="No learning")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Fold {0} | Precision-recall curve for classes: "+ ', '.join(labels).format(fold))
    plt.legend(loc="lower right")
    plt.savefig('PR_fold_{0}.tiff'.format(fold), format="tiff", dpi=600)
    plt.show()
    plt.close('all')
    


img_width = img_height = 512
batch_size = 2
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
fold_ = 0
histories = []
predictions = []
ground_truth = []
pred_prob = []
all_cms = []
preds_raw = []
all_val_gt_cat = []

# patient_IDs = image_df['patient_ID'].drop_duplicates()

#If you have a dedicated test set, comment below line and read the test set
train_patient_class_dict = pd.Series(image_df.class_names.values,index=image_df.patient_ID).to_dict()
train_patient_IDs = image_df['patient_ID'].drop_duplicates()

for train_index, _ in skf.split(list(train_patient_class_dict.keys()), list(train_patient_class_dict.values())):
    #Split the dataframe into train and test
    train_patients = [train_patient_IDs.values[i] for i in train_index]
    trainData = image_df[image_df['patient_ID'].isin(train_patients)]
    valData = image_df[~image_df['patient_ID'].isin(train_patients)]
    print('Initializing Kfold %s'%str(fold_))
    print('Train shape:',trainData.shape)
    print('Val shape:',valData.shape)
    
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

    
    # Create a model with three classes
    
    model = create_model(img_size=512,classes=3)
    
    STEP_SIZE_TRAIN = int(np.ceil(train_generator.n/train_generator.batch_size))
    STEP_SIZE_VALID = int(np.ceil(validation_generator.n/validation_generator.batch_size))
    
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
    pred = model.predict_generator(validation_generator,steps = STEP_SIZE_VALID,verbose=1)
    pred_prob.append(np.max(pred,axis=1))
    #Get the predicted class
    predicted_class_indices=np.argmax(pred,axis=1)
    # Get the groun truth
    labels=validation_generator.classes
    
    class_names = list(train_generator.class_indices.keys())
    
    # Save predictions and groundtruth in thier respective lists
    predictions.append(predicted_class_indices)
    ground_truth.append(labels)
    fold_+=1
    
    
    # Confusion matrix on validation data
    cm = plot_cm(labels, predicted_class_indices, fold_, class_names)
    all_cms.append(cm)

    # Save predictions in dataframe
    val_pred_df = get_test_predictions(valData) 
    val_pred_df.to_csv('Val_prediction_fold{0}.csv'.format(fold_), index=False)
    # print(val_pred_df.to_markdown())

    # Save metrics in dataframe
    val_metrics_df = get_metrics(labels, predicted_class_indices)
    val_metrics_df.to_csv('Val_metrics_fold{0}.csv'.format(fold_), index=False)
    print(val_metrics_df.to_markdown())

    # Plot curves
    
    val_gt_cat = to_categorical(labels)
    all_val_gt_cat.extend(val_gt_cat)
    get_roc_curve(class_names,pred, val_gt_cat, fold_)
    get_pr_curve(class_names,pred, val_gt_cat, fold_)
    
    
# Get average cm
get_average_cm(all_cms, class_names)

#Get average metric plots
get_roc_curve(class_names,preds_raw, all_val_gt_cat, 'Average')
get_pr_curve(class_names,preds_raw, all_val_gt_cat, 'Average')

#Training and validation curves
plot_histories(histories)