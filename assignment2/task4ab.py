import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
from task2 import SoftmaxTrainer, calculate_accuracy
from trainer import BaseTrainer
np.random.seed(0)

if __name__ == '__main__':
    # hyperparameters
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Settings for task 3. Keeping them for task 4ab
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # model with 64 in hidden layer
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # Task 4a: model with 32 in hidden layer
    neurons_per_layer = [32, 10]

    model_a = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_a = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_a, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_a, val_history_a = trainer_a.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model_a.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model_a.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model_a))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model_a))

    # Task 4b: model with 128 in hidden layer
    neurons_per_layer = [128, 10]

    model_b = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_b = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_b, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_b, val_history_b = trainer_b.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model_b.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model_b.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model_b))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model_b))

    # Plot loss for the three models
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    #utils.plot_loss(train_history["loss"], "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss (64 nodes)")
    utils.plot_loss(val_history_a["loss"], "Validation Loss (32 nodes)")
    utils.plot_loss(val_history_b["loss"], "Validation Loss (128 nodes)")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.90, .99])
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy (64 nodes)")
    utils.plot_loss(val_history_a["accuracy"], "Validation Accuracy (32 nodes)")
    utils.plot_loss(val_history_b["accuracy"], "Validation Accuracy (128 nodes)")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4ab_val_loss_and_accuracy.png")



