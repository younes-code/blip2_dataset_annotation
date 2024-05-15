# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import re

# def visualize_embeddings(embeddings, classes):
#     # Reduce dimensionality using PCA
#     reduced_embeddings = PCA(n_components=2).fit_transform(embeddings)
#     # Define unique colors for each class
#     unique_classes = np.unique(classes)
#     colors = plt.cm.get_cmap('tab20', len(unique_classes))
#     color_dict = {class_name: colors(i) for i, class_name in enumerate(unique_classes)}
#     # Plot the reduced embeddings with unique colors for each class
#     plt.figure(figsize=(10, 8))
#     for class_name in unique_classes:
#         indices = [i for i, label in enumerate(classes) if label == class_name]
#         plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=class_name, c=[color_dict[class_name]])
#     plt.title('Embeddings Visualization')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.legend()
#     plt.show()

# def extract_captions_and_classes(embeddings_dict):
#     captions = []
#     classes = []
#     for video_name, embeddings in embeddings_dict.items():
#         for caption_embeddings in embeddings:
#             # Ensure caption_embeddings is a 2D array
#             caption_embeddings = np.atleast_2d(caption_embeddings)
#             # Add caption embedding to the list
#             captions.append(caption_embeddings[0])
#             # Extract class from video name using regex
#             class_name = re.match(r'^([^0-9]+)', video_name).group(1)
#             classes.append(class_name)
#     return np.array(captions), np.array(classes)

# if __name__ == '__main__':
#     # Load embeddings from the file
#     data = np.load('captions_embeddings.npz', allow_pickle=True)
#     embeddings_dict = {key: value for key, value in data.items()}

#     # Extract captions and classes from embeddings
#     captions, classes = extract_captions_and_classes(embeddings_dict)

#     # Convert lists to numpy arrays
#     captions = np.array(captions)
#     classes = np.array(classes)

#     # Perform dimensionality reduction using PCA
#     reduced_embeddings = PCA(n_components=2).fit_transform(captions)

#     # Visualize the embeddings
#     visualize_embeddings(reduced_embeddings, classes)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import re

def visualize_embeddings(embeddings, classes, filter_classes=None):
    if filter_classes is not None:
        # Filter embeddings and classes based on filter_classes
        filter_indices = np.isin(classes, filter_classes)
        embeddings = embeddings[filter_indices]
        classes = classes[filter_indices]
    
    # Reduce dimensionality using LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    reduced_embeddings = lda.fit_transform(embeddings, classes)
    # Define unique colors for each class
    unique_classes = np.unique(classes)
    colors = plt.cm.get_cmap('tab20', len(unique_classes))
    color_dict = {class_name: colors(i) for i, class_name in enumerate(unique_classes)}
    # Plot the reduced embeddings with unique colors for each class
    plt.figure(figsize=(10, 8))
    for class_name in unique_classes:
        indices = [i for i, label in enumerate(classes) if label == class_name]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=class_name, c=[color_dict[class_name]])
    plt.title('Embeddings Visualization using LDA')
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.legend()
    plt.show()

def extract_captions_and_classes(embeddings_dict):
    captions = []
    classes = []
    for video_name, embeddings in embeddings_dict.items():
        for caption_embeddings in embeddings:
            # Ensure caption_embeddings is a 2D array
            caption_embeddings = np.atleast_2d(caption_embeddings)
            # Add caption embedding to the list
            captions.append(caption_embeddings[0])
            # Extract class from video name using regex
            class_name = re.match(r'^([^0-9]+)', video_name).group(1)
            classes.append(class_name)
    return np.array(captions), np.array(classes)

if __name__ == '__main__':
    # Load embeddings from the file
    data = np.load('captions_embeddings.npz', allow_pickle=True)
    embeddings_dict = {key: value for key, value in data.items()}

    # Extract captions and classes from embeddings
    captions, classes = extract_captions_and_classes(embeddings_dict)

    # Convert lists to numpy arrays
    captions = np.array(captions)
    classes = np.array(classes)

    # Specify classes to be plotted (optional)
    filter_classes = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Normal_Videos_", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    # filter_classes=['Abuse','RoadAccidents','Normal_Videos_','Burglary','Explosion','Robbery',"Arrest"]
    # Visualize the embeddings using LDA with optional class filtering
    visualize_embeddings(captions, classes, filter_classes)