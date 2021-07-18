from autoencoder import VAE
import numpy as np
import os
import boto3
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
  
    # data directories
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    parser.add_argument('--output', type=str, default=os.environ.get('SM_CHANNEL_OUTPUT'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        
    # model directory: SageMaker default is /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


def get_train_data(data_dir):
    x_train = []
    for root, _, file_names in os.walk(data_dir):
        for file in file_names:
            file_path = os.path.join(root, file)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
    # cast to numpy format for tensorflow
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # 3000 samples -> (3000, 256, 64, 1)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 64, 1),
        conv_filters=[512, 256, 128, 64, 32],
        conv_kernels=[3, 3, 3, 3, 3],
        conv_stride=[2, 2, 2, 2, (2, 1)],
        latent_space_dim=128  # 128 neurons as output dim
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    
    print("loading training data...")
    args, _ = parse_args()

    x_train = get_train_data(args.data)
    
    autoencoder = train(x_train, args.learning_rate, args.batch_size, args.epochs)
    
    print("saving model...")
    autoencoder.save(args.model_dir)
