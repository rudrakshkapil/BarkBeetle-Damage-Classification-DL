# Main training and testing file for carrying out experiments. 

import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from augment_data import augment_data, get_options_dir
from crop_data import crop_images, read_data, save_image_crops
from df_repo.deepforest import main, model
from df_repo.deepforest import evaluate
from df_repo.deepforest import main
from df_repo.deepforest import visualize
from pytorch_lightning import loggers as pl_loggers
import numpy as np
from split_data import overwrite_directory, split_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from visualize import convert_images


def get_labels(filename):
    '''
    Get y_actual values from info file.
    Input: filename
    Output: list of labels
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split(",") for line in lines]
        labels = [line[-1] for line in lines]

    return labels[1:]

def get_color_model(exp_params, augmemnt_params):
    '''
    Function to get color detection model. Backbone and FPN pre-trained weights are loaded and frozen.
    Input: exp_params, augment_params -- dictionaries for passing values to model.config
    Ouput: DeepForest model object (modified RetinaNet)
    '''

    policy = augmemnt_params.get('policy')
    options = augmemnt_params.get('options')

    # dataset directories (default none is without augmentation)
    train_dir= "split/train" if policy =='none' else f"augmented/{policy}/{get_options_dir(options)}"
    val_dir = f"split/val"

    # define new color detection model
    color_model = main.deepforest(num_classes=4, label_dict={"green":0,"yellow":1,"red":2,"leafless":3})

    # get pretrained model
    deepforest_release_model = main.deepforest()
    deepforest_release_model.use_release()

    # load backbone and box prediction portions
    color_model.model.backbone.load_state_dict(deepforest_release_model.model.backbone.state_dict())
    color_model.model.head.regression_head.load_state_dict(deepforest_release_model.model.head.regression_head.state_dict())
    color_model.model.backbone.requires_grad_(False)
    color_model.model.head.regression_head.requires_grad_(False)

    # set new training parameters
    color_model.config["gpus"] = '-1' 
    color_model.config["score_threshold"] = 0
    color_model.config["optim"] = exp_params['optimizer']
    color_model.config['batch_size'] = exp_params['batch_size']
    color_model.config["train"]["epochs"] = exp_params['epochs']
    color_model.config["train"]["lr"] = exp_params['lr']
    color_model.config["train"]["csv_file"] = f"./datasets/{train_dir}/_train_info.csv" 
    color_model.config["train"]["root_dir"] = f"./datasets/{train_dir}"
    color_model.config["train"]["fast_dev_run"] = False 
        
    # set validation parameters as well
    color_model.config["validation"]["csv_file"] = f"./datasets/{val_dir}/_val_info.csv"
    color_model.config["validation"]["root_dir"] = f"./datasets/{val_dir}"
    color_model.config["validation"]["val_accuracy_interval"] = 5

    # return
    return color_model


def prepare_square_data(flight, augment_params):
    '''
    Function to crop squares from images and split into train-val-test for passed flight.
    Input:
        flight: which flight data to work on
        augment_params: what augmentation to carry out. 
    '''
    # crop data
    scale = 60 if flight == 'jun60' else 50
    annotations = read_data(f'bb_repo/annotations/{flight}.raw')
    img_crops = crop_images(annotations, f'datasets/mosaics/{flight}.png', scale)
    save_image_crops(f'datasets/cropped_squares', img_crops, flight, overwrite=True)

    # data splitting
    np.random.seed(0)
    split_data({"crop_dir": f"datasets/cropped_squares"})

    # augment data if needed
    augment_data(augment_params)


def train_all_flights(exp_params):
    '''
    Utility function to train given experiment over all flights and validate performance.
    Input: exp_params -- dictionary to denote current experiment parameters.
    Output: Mean Accuracy over all flights.
    '''
    # skip if training not needed
    if not exp_params['training']:
        return

    # get experiment parameters
    network = exp_params.get('network')
    mask_shape = exp_params.get('mask_shape')
    optimizer = exp_params.get('optimizer')
    augment_params = exp_params.get('augment_params')
    augment_policy = augment_params.get('policy')
    augment_options = augment_params.get('options')
    loss = exp_params.get('loss')
    batch_size = exp_params.get('batch_size')
    epochs = exp_params.get('epochs')
    lr = exp_params.get('lr') 

    # loop training steps over flights individually
    accs = []
    flights = ['jun60','jul90','jul100','aug90','aug100']
    for flight in flights:
        # get directory name
        if augment_policy == 'none':
            exp_name = f"nw={network}/shape={mask_shape}_opt={optimizer}_aug={augment_policy}/loss={loss}_bs={batch_size}_epochs={epochs}_lr={lr}/flight={flight}"
        else:
            exp_name = f"nw={network}/shape={mask_shape}_opt={optimizer}_aug={augment_policy}/{get_options_dir(augment_options)}/loss={loss}_bs={batch_size}_epochs={epochs}_lr={lr}/flight={flight}"
        
        # make directory if needed
        try:
            os.makedirs(f'experiments/{exp_name}')
        except:
            choice_str = "Experiment folder exists already and trying to train. Overwrite? [Y/N] => "
            if not network == 'deepforest' or input(choice_str) in 'yY':
                overwrite_directory(f'experiments/{exp_name}')
            else:
                print(f"Skipping {flight}")
                continue

        # prepare data directories
        prepare_square_data(flight, augment_params)

        # deep learning model
        if exp_params['network'] == 'deepforest':
            # define new color detection model
            color_model = get_color_model(exp_params, augment_params)

            # run trainer
            tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{exp_name}/logs")
            color_model.create_trainer(logger=tb_logger)
            color_model.trainer.fit(color_model)

            # save model
            color_model.save_model(f'./experiments/{exp_name}/model')

            # evaluate for progress
            val_csv = f"./datasets/split/val/_val_info.csv"
            val_dir = os.path.dirname(val_csv)
            predictions = color_model.predict_file(csv_file=val_csv, root_dir=val_dir)

            # accuracy
            y_preds = predictions['label']
            y_actual = get_labels(val_csv)
            print(f"Accuracy for {flight} = {np.mean(y_preds == y_actual)*100}%")

        # classical ML networks
        else:
            # data directories
            train_dir= "split/train" if augment_policy =='none' else f"augmented/{augment_policy}/{get_options_dir(augment_options)}"
            train_names = os.listdir(f"./datasets/{train_dir}")
            train_names.remove('_train_info.csv')
            train_csv = f"./datasets/{train_dir}/_train_info.csv"

            val_dir = f"split/val"
            val_names = os.listdir(f"./datasets/{val_dir}")
            val_names.remove('_val_info.csv')
            val_csv = f"./datasets/{val_dir}/_val_info.csv"

            # load train and val data
            y_train = get_labels(train_csv)
            X_train = []
            for name in train_names:
                X_train.append(np.array(Image.open(f'./datasets/{train_dir}/{name}')))
            X_train, _ = convert_images(X_train, color_space=exp_params['color_space'])
            X_train = np.asarray(X_train)

            y_val = get_labels(val_csv)
            X_val = []
            for name in val_names:
                X_val.append(np.array(Image.open(f'./datasets/{val_dir}/{name}')))
            X_val, _ = convert_images(X_val, color_space=exp_params['color_space'])
            X_val = np.asarray(X_val)

            # flatten X and scale
            h,w,c = X_train[0].shape
            X_train = X_train.reshape(-1, h*w*c) 
            # X_train = pca_50.fit_transform(X_train)
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)

            X_val = X_val.reshape(-1, h*w*c)
            # X_val = pca_50.fit_transform(X_val)
            scaler = MinMaxScaler()
            X_val = scaler.fit_transform(X_val)

            # create and train SVM classifier
            if exp_params['network'] == 'SVM':
                reg = exp_params['svm_params']['reg']
                kernel = exp_params['svm_params']['kernel']
                degree = exp_params['svm_params']['degree']
                gamma = exp_params['svm_params']['gamma']
                tol = exp_params['svm_params']['tol']
                color_model = SVC(C=reg, kernel=kernel, degree=degree, gamma=gamma, tol=tol, decision_function_shape='ovo', max_iter=100000)
                color_model.fit(X_train, y_train)

            # create and train randforest classifier
            if network == 'randforest':
                color_model = RandomForestClassifier(criterion='gini')
                color_model.fit(X_train, y_train)

            if network == 'KNN':
                color_model = KNeighborsClassifier(n_neighbors=20, algorithm='kd_tree')
                color_model.fit(X_train, y_train)

             # save model
            dump(color_model, f'./experiments/{exp_name}/model')

            # print accuracy
            y_pred = color_model.predict(X_val)
            acc = np.mean(y_pred == y_val)*100
            print(f"Accuracy for {flight} = {acc}%")
            accs.append(acc)


    if len(accs):
        return np.mean(accs)




    
def validate_all_flights(exp_params, evaluate_other_flight=None):
    '''
    Utility function to validate given experiment over all flights.
    Input: 
        Exp_params: dictionary for parameters
        evaluate_other_flight: bool that if True, evaluates the model trained on one flight data on all other flight data.
    '''
    # get experiment parameters
    network = exp_params.get('network')
    mask_shape = exp_params.get('mask_shape')
    optimizer = exp_params.get('optimizer')
    augment_params = exp_params.get('augment_params')
    augment_policy = augment_params.get('policy')
    augment_options = augment_params.get('options')
    loss = exp_params.get('loss')
    batch_size = exp_params.get('batch_size')
    epochs = exp_params.get('epochs')
    lr = exp_params.get('lr') 
    test_or_val = exp_params.get('test_or_val', 'val')
    labels = ['Green', 'Yellow', 'Red', 'Leafless']

    # get directory name
    if augment_policy == 'none':
        exp_name = f"nw={network}/shape={mask_shape}_opt={optimizer}_aug={augment_policy}/loss={loss}_bs={batch_size}_epochs={epochs}_lr={lr}"
    else:
        exp_name = f"nw={network}/shape={mask_shape}_opt={optimizer}_aug={augment_policy}/{get_options_dir(augment_options)}/loss={loss}_bs={batch_size}_epochs={epochs}_lr={lr}"
    parent_dir = exp_name

    # file to write to
    if evaluate_other_flight is None:
        output_file = open(f'./experiments/{exp_name}/results.txt', 'w')
    else:
        output_file = open(f'./experiments/{exp_name}/results_of_{evaluate_other_flight}_model.txt', 'w')

    # loop training steps over flights individually
    accuracies = []
    y_preds = []
    y_actuals = []
    flights = ['jun60','jul90','aug100']
    for flight in flights:
        # dataset directories (default none is without augmentation)
        output_file.write(f'\n\nFlight: {flight}\n')
        val_dir = f"split/val"

        # get directory name
        exp_name = f"{parent_dir}/flight={flight}"

        # prepare data (no augmentation of validation/test set so both false)
        #flight_param = flight if evaluate_other_flight is None else evaluate_other_flight
        prepare_square_data(flight, augment_params)

        if network == 'deepforest':
            # load color detection model
            if evaluate_other_flight is None:
                color_model = main.deepforest.load_from_checkpoint(f'./experiments/{exp_name}/model')
            else: 
                model_loc = f"./experiments/{parent_dir}/flight={evaluate_other_flight}/model"
                color_model = main.deepforest.load_from_checkpoint(model_loc)
    
            # evaluate model on test/val set
            val_csv = f"./datasets/split/{test_or_val}/_{test_or_val}_info.csv"
            val_dir = os.path.dirname(val_csv)
            predictions = color_model.predict_file(csv_file=val_csv, root_dir=val_dir)

            # extract predictions
            y_pred = predictions['label']

        # traditional ML
        else:

            # data directories
            test_dir = f"split/{test_or_val}"

            # load data
            test_names = os.listdir(f"./datasets/{test_dir}")
            test_names.remove('_test_info.csv')
            X_test = []
            for name in test_names:
                X_test.append(np.array(Image.open(f'./datasets/{test_dir}/{name}')))
            X_test = np.asarray(X_test)

            # load and evaluate SVM 
            if exp_params['network'] == 'SVM':
                color_model = load(f'./experiments/{exp_name}/model')

                # predict (test/val accuracy)
                h,w,c = X_test[0].shape
                X_test = X_test.reshape(-1, h*w*c)
                scaler = MinMaxScaler()
                X_test = scaler.fit_transform(X_test)
                y_pred = color_model.predict(X_test)
        
        # accuracy
        val_csv = f"./datasets/split/{test_or_val}/_{test_or_val}_info.csv"
        y_actual = get_labels(val_csv)
        accuracies.append(np.mean(y_pred == y_actual)*100)

        # confusion matrix
        y_preds.append(y_pred)
        y_actuals.append(y_actual)
        plt.figure(figsize=(20, 20), dpi=120)
        ConfusionMatrixDisplay.from_predictions(y_pred=y_pred, y_true=y_actual, display_labels=labels, cmap="plasma")
        if not evaluate_other_flight:
            plt.savefig(f'./experiments/{exp_name}/{flight}_cfm.png')

        # print which ones are wrong
        if network == 'deepforest':
            ground_truth = pd.read_csv(val_csv)
            result = evaluate.evaluate(predictions=predictions, ground_df=ground_truth,root_dir= val_dir, savedir=None)     
            output_file.write(result["class_recall"].__str__())
            output_file.write(f"\nIncorrect predictions for {flight}: ")
            output_file.write(result['results'].loc[result['results'].predicted_label != result['results'].true_label].__str__())

    # accuracies
    output_file.write(f"\nAccuracies => {accuracies}")
    output_file.write(f"\nMean Accuracy = {np.mean(accuracies)}%")

    # combined confusion matrix
    f, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=120, sharey='row')
    for idx in range(len(y_preds)):
        cfm = confusion_matrix(y_actuals[idx], y_preds[idx])
        disp = ConfusionMatrixDisplay(cfm, display_labels=labels)
        disp.plot(ax=axes[idx], xticks_rotation=30, cmap='plasma')
        disp.ax_.set_title(flights[idx])
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if idx!=0:
            disp.ax_.set_ylabel('')

    exp_name = exp_name.split('/')[:-1] # or just set exp_name = parent_dir
    exp_name = '/'.join(exp_name)
    f.text(0.4, 0.04, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0)
    f.colorbar(disp.im_, ax=axes)
    combined_cfm_name = "combined_cfm" if not evaluate_other_flight else f"combined_cfm_from_{evaluate_other_flight}_model"
    plt.savefig(f'./experiments/{exp_name}/{combined_cfm_name}.pdf')

    # close output file
    output_file.close()
    print(f"Wrote results to results.txt in {exp_name}")



# multiprocessing => needs to be enclosed in __main__
if __name__ == "__main__":

    # Experiment details
    exp_params = {'network': 'deepforest',
        'mask_shape': 'square',
        'optimizer': 'adamw', 
        'augment_params': {'policy': 'custom', 
                           'options': {'flip': False,
                                       'rotate': False,
                                       'crop': False,
                                       'jitter': False,
                                       'warp': True,
                                       'blur': False,
                                       }
                          },
        'svm_params': {'reg':10, 'kernel': 'rbf', 'gamma':0.5, 'degree':10, 'tol':1e-9},
        'color_space': 'YUV',
        'loss': 'None',
        'batch_size': 2, 
        'training': True,
        'epochs': 50,
        'lr': 0.00001, # 1e-5 with adamw no augm on aug100 for 40 epochs and bs 2 gave best performance so far ~96%
        'test_or_val': 'test',
    }

    #train_all_flights(exp_params)
    validate_all_flights(exp_params, evaluate_other_flight=None)


    # rf aug=rcwb, gini => 40.238
    # svm => 53.095 
    # KNN no aug kdtree n=50 => 53.095

    # # testing effect of batch size (left: optim, augmentation first)
    # degrees = [5,6,7,8,9,10]
    # regs = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    # tols = [1e-5,1e-6,1e-7,1e-8]

    # best = 0
    # best_choice = ""
    # for degree in degrees:
    #     for reg in regs:
    #         for tol in tols:
    #             exp_params['svm_params']['reg'] = reg
    #             exp_params['svm_params']['tol'] = tol
    #             exp_params['svm_params']['degree'] = degree
    
    #             acc = train_all_flights(exp_params)
    #             print(acc)
    #             #validate_all_flights(exp_params)

    #             if acc > best:
    #                 best = acc
    #                 best_choice = f"New best => {best} => d={degree}, r={reg}, t={tol}"
    #                 print(best_choice)
    


    # inspect augmented data
    # prepare_square_data('jun60', True, False)



    


    

