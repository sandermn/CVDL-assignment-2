import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    print("Training standard model:\n")
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
    print("\n\n")

    # Example created in assignment text - Comparing with and without shuffling.
    # YOU CAN DELETE EVERYTHING BELOW!

    # model with improved sigmoid
    use_improved_sigmoid = True

    print("Training model with improved sigmoid:\n")

    model_is = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_is = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_is, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_is, val_history_is = trainer_is.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
        cross_entropy_loss(Y_train, model_is.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
        cross_entropy_loss(Y_val, model_is.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model_is))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model_is))
    print("\n\n")

    # model with improved weight init
    use_improved_weight_init = True

    print("Training model with improved weight init:\n")
    
    model_iw = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_iw = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_iw, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_iw, val_history_iw = trainer_iw.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
        cross_entropy_loss(Y_train, model_iw.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
        cross_entropy_loss(Y_val, model_iw.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model_iw))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model_iw))
    print("\n\n")
    

    # model with momentum
    use_momentum = True
    learning_rate = 0.02

    print("Training model with momentum:\n")
    model_um = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_um = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_um, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_um, val_history_um = trainer_um.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
        cross_entropy_loss(Y_train, model_um.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
        cross_entropy_loss(Y_val, model_um.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model_um))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model_um))
    print("\n\n")

    # plotting the comparison
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0, .5])
    utils.plot_loss(train_history["loss"], "Task 3 Model", npoints_to_average=10)
    utils.plot_loss(train_history_is["loss"], "Task 3 Model - with improved sigmoid", npoints_to_average=10)
    utils.plot_loss(train_history_iw["loss"], "Task 3 Model - with improved weight init", npoints_to_average=10)
    utils.plot_loss(train_history_um["loss"], "Task 3 Model - with momentum", npoints_to_average=10)
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .99])
    utils.plot_loss(val_history["accuracy"], "Task 3 Model")
    utils.plot_loss(val_history_is["accuracy"], "Task 3 Model - with improved sigmoid", npoints_to_average=10)
    utils.plot_loss(val_history_iw["accuracy"], "Task 3 Model - with improved weight init", npoints_to_average=10)
    utils.plot_loss(val_history_um["accuracy"], "Task 3 Model - with momentum", npoints_to_average=10)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.savefig("task3_loss_and_val.png")











    """
    shuffle_data = False
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)
    shuffle_data = True

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Task 2 Model - No dataset shuffling", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Task 2 Model - No Dataset Shuffling")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
    """
