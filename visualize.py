# Script to plot t-sne, scatter, histograms

import pandas as pd
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
from augment_data import augment_data, get_options_dir
from crop_data import crop_all_data
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import cv2

from split_data import split_data



def load_data(path):
    '''
    Loads and returns images & labels.
    Input: path containing directories that contain images of each label
    Output: X (flattened images), y (labels)
    '''
    X, y = [], []

    # loop through all images 
    image_names = os.listdir(path)
    X = [cv2.imread(f"{path}/{name}").flatten() for name in image_names if name.endswith('.png')]
    y = [name.split('_')[1] for name in image_names if name.endswith('.png')]

    return X, y

def convert_images(label_images, color_space):
    '''
    Convert images from BGR to passed color_space.
    Input: 
        label_images: list of cv2 BGR images
        color_space: target color spcae for conversion
    Output:
        converted list and string for pyplot colors
    '''
    # conversions
    if color_space == 'BGR':
        color = 'bgr'
    if color_space == 'YUV':
        color = 'ygm'
        label_images = [cv2.cvtColor(img, cv2.COLOR_BGR2YUV) for img in label_images]
    if color_space == 'LAB':
        color = 'cmy'
        label_images = [cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for img in label_images]
    if color_space == 'HSV':
        color = 'rbm'
        label_images = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in label_images]
    if color_space == 'HLS':
        color = 'ybr'
        label_images = [cv2.cvtColor(img, cv2.COLOR_BGR2HLS) for img in label_images]
    if color_space == 'YCbCr':
        color = 'ybr'
        label_images = [cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in label_images]
    if color_space == 'LUV':
        color = 'cbm'
        label_images = [cv2.cvtColor(img, cv2.COLOR_BGR2LUV) for img in label_images]
    return label_images, color


def average_histograms(color_space='BGR'):
    '''
    Calculates and plots histograms for each channel -- averaged over class label
    Input: color space of images
    Output: None (saves figure)
    '''
    # extract all cropped data to cropped_squares directory
    crop_all_data()
    cropped_dir = './datasets/cropped_squares'

    # loop through class labels
    max_freq = 0
    fig, axes = plt.subplots(1,2, figsize=(10,4), dpi=60)
    labels = ['yellow', 'red']
    for idx,label in enumerate(labels):
        # load images
        label_dir = f"{cropped_dir}/{label}"
        label_names = os.listdir(label_dir)
        label_images = [cv2.imread(f"{label_dir}/{name}", ) for name in label_names]

        # convert
        label_images, color = convert_images(label_images, color_space)
  
        # compute and plot the image histograms
        axes[idx].set_title(f"{label.capitalize()} Attack Stage")
        for i,(color,label) in enumerate(zip(color, color_space)):
            hist = cv2.calcHist(label_images,[i],None,[256],[0,256])
            if color_space == 'YCbCr':
                if i == 1: 
                    label = 'Cb'
                elif i == 2:
                    label = 'Cr'

            axes[idx].plot(hist, color=color, label=label)
            if max(hist) > max_freq:
                max_freq = max(hist)
        axes[idx].legend()
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Frequency")

    plt.subplots_adjust(wspace=0.20, hspace=0.3)
    plt.setp(axes, ylim=(0,max_freq*1.1))
    color_space = 'RGB' if color_space == 'BGR' else color_space
    color_space = 'CIELAB' if color_space == 'LAB' else color_space
    fig.suptitle(f"{color_space} Color Space Histograms for Dataset Images by Class")
    plt.savefig(f'./plots/histograms/({color_space}) average histograms by class.pdf')




def plot_all_images_in_color_space(color_space):
    '''
    Function to plot each image in the passed color space. 
    3 coordinates for an image are obtained by averaging across each channel. 
    Input: color_space
    Output: None
    '''
    # extract all cropped data to cropped_squares directory
    crop_all_data()
    cropped_dir = './datasets/cropped_squares'

    # plot labels
    fig = plt.figure(figsize=(12,6), dpi=120)
    for idx in range(1):
        ax = fig.add_subplot(111+idx,projection='3d')
        ax.set_xlabel(color_space[0])
        ax.set_ylabel(color_space[1])
        ax.set_zlabel(color_space[2])
        if color_space == 'YCbCr':
            ax.set_ylabel(color_space[1:3])
            ax.set_zlabel(color_space[3:5])

        # loop through class labels
        labels = ['green', 'yellow', 'red', 'leafless']
        for idx,label in enumerate(labels):
            # load images
            label_dir = f"{cropped_dir}/{label}"
            label_names = os.listdir(label_dir)
            label_images = [cv2.imread(f"{label_dir}/{name}", ) for name in label_names]

            # convert to color space
            label_images, _ = convert_images(label_images, color_space)

            # average each image by channel
            label_images = np.asarray([np.around(np.mean(image, axis=(0,1)), 2) for image in label_images])
            
            # print(label_images.shape)
            # # PCA
            # pca_50 = PCA(n_components=30)
            # label_images = pca_50.fit_transform(label_images.reshape(len(label_images),-1)) 

            # plot 
            color = f'{label}' if not label == 'leafless' else 'gray'
            for i in range(len(label_names)):
                if np.sum(label_images[i]) == 0:
                    continue
                #if label in 'greenred':
                ax.scatter(label_images[i,0], label_images[i,1], label_images[i,2], c=color, label=label, zorder=np.random.randint(0,100), alpha=0.5)
                #if label_names[i] in ['aug90_green_263.png', 'aug90_red_147.png', 'aug100_leafless_120.png', 'aug100_yellow_152.png']:
                #if color == 'green':
                #    ax.text(label_images[i,0], label_images[i,1], label_images[i,2],  '%s' % (str(i)), size=5, zorder=1000,  color='k') 

    fig.suptitle(f'Average Color Space ({color_space}) Position of Each Image\nSame Plot from Different Viewpoints')
    #plt.savefig(f'./plots/3D_scatter/({color_space}) average histograms by class.pdf')
    plt.show() # should screenshot manually



def generate_all_histograms_and_3D_plots():
    '''
    Function to loop over all color spaces and plot histograms and t-sne
    '''
    options = ['BGR', 'YUV', 'LAB', 'HSV', 'HLS', 'YCbCr', 'LUV']
    for option in options:
        #average_histograms(color_space=option)
        plot_all_images_in_color_space(color_space=option)



def tsne_visualize(X, y, augment_params):
    '''
    Visualize data with TSNE 
    Input: X, y -- data
    Output: None -- saves to plots directory
    '''

    # PCA for dimensionality reduction
    #np.random.seed(0)
    print("Performing PCA...")
    pca_50 = PCA(n_components=30)
    pca_result_50 = pca_50.fit_transform(X)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

    # perform t-sne
    tsne = TSNE(n_components=2, verbose=1, random_state=0, perplexity=35)
    z = tsne.fit_transform(pca_result_50) 

    # extract data 
    df = pd.DataFrame()
    df["y"] = y
    df["Dimension 1"] = z[:,0]
    df["Dimension 2"] = z[:,1]

    # determine name
    name = f"T-SNE Plot of Bark Beetle Data"
    save_name = f"aug={augment_params['policy']}"
    if not augment_params['policy'] == 'none' :
        name += " (Color Jittered)"
        save_name += f"_{get_options_dir(augment_params['options'])}"

    # plot and save figure
    sns.scatterplot(x="Dimension 1", y="Dimension 2", hue=df.y.tolist(),
                palette=['green', 'gray', 'red', 'orange'],
                data=df).set(title=name)         
    plt.savefig(f"./plots/t-sne/{save_name}.pdf")
    plt.show()


def tsne_vis_utility(augment_params):
    '''
    Wrapper function for t-sne visualization call.
    Combines all splits of data first before calling. 
    '''

    # crop data
    crop_all_data()
    
    # data splitting
    np.random.seed(0)
    split_data({"crop_dir": f"datasets/cropped_squares"})

    # augment data if needed
    augment_data(augment_params)

    # load data and combine
    X, y = [], []
    train_dir= "split/train" if augment_params['policy']=='none' else f"augmented/{augment_params['policy']}/{get_options_dir(augment_params['options'])}"
    X_train, y_train = load_data(f'./datasets/{train_dir}')
    X.extend(X_train)
    y.extend(y_train)
    X_val, y_val = load_data(f'./datasets/split/val')
    X.extend(X_val)
    y.extend(y_val)
    X_test, y_test = load_data(f'./datasets/split/test')
    X.extend(X_test)
    y.extend(y_test)

    # run T-SNE
    tsne_visualize(X, y, augment_params)



if __name__ == "__main__":
    # naming details
    # augmentation = 'none'
    # name = f"Bark Beetle Data T-SNE projection (Augmentation - {augmentation}, Loss - {loss})"

    ## histogram visualization
    #generate_all_histograms_and_3D_plots()
    augment_params = {'policy': 'custom', 
                      'options': {'flip': False,
                                   'rotate': False,
                                   'crop': False,
                                   'jitter': True,
                                   'warp': False,
                                   'blur': False,
                                   }
                     }
    #tsne_vis_utility(augment_params)
    generate_all_histograms_and_3D_plots()