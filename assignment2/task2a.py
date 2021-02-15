import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)

    X = (X - X.mean())/X.std()
    X = np.insert(X, 784, 1, axis=1)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)

    cel =  (-np.sum(targets * np.log(outputs), axis=1))
    cel = (1 / targets.shape[0]) * np.sum(cel)
    return cel



class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.hidden_layer_ouput = [None, None]

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For peforming the backward pass, you can save intermediate activations in varialbes in the forward pass.
        # such as self.hidden_layer_ouput = ...

        # sigmoid activation for the first layers
        self.hidden_layer_ouput[0] = 1/(1+np.exp(np.transpose(-self.ws[0]).dot(X.T)))
        #self.hidden_layer_ouput[0] = 1/(1+np.exp(-X.dot(self.ws[0])))
        #print(f' HLO_0 = {self.hidden_layer_ouput[0].shape}')

        # softmax activation for the last hidden layer
        self.hidden_layer_ouput[1] = np.divide(np.exp(self.hidden_layer_ouput[0].T.dot(self.ws[1])),np.transpose(np.array([np.sum(np.exp(self.hidden_layer_ouput[0].T.dot(self.ws[1])), axis=1)])))
        
        #print(self.ws[1].shape, self.hidden_layer_ouput[0].shape)

        #zk = self.hidden_layer_ouput[0].T.dot(self.ws[1])
        #self.hidden_layer_ouput[1] = (np.exp(zk) / np.sum(zk))
        #print(self.hidden_layer_ouput[1].shape)

        #print(f'HLO_1 = {self.hidden_layer_ouput[1].shape}')



        return self.hidden_layer_ouput[1]


    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        """
        self.grads = []
        self.grads.append(np.zeros((785, 64)))
        self.grads.append(np.zeros((64, 10)))

        for i, (grad, w) in enumerate(zip(self.grads, self.ws)):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
        
        self.grads[1] = (1 / targets.shape[0])*(np.transpose(self.hidden_layer_ouput).dot(-(targets - outputs)))
        self.grads[0] = (1 / targets.shape[0])*(np.transpose(X).dot(-(targets - outputs)))
        """
        samples = targets.shape[0]
        w_ji = self.ws[0]
        w_kj = self.ws[1]
        zj = X.dot(w_ji)
        dk = -(targets-outputs)
        fderv = (1/(1+np.exp(-zj))*(1-1/(1+np.exp(-zj))))

        #print(self.hidden_layer_ouput[1].T.shape, dk.shape)
        
        self.grads[1] = 1/samples*(self.hidden_layer_ouput[0]).dot(dk)
        
        dj = fderv * (dk.dot(w_kj.T))

        self.grads[0] = 1/samples*(X.T.dot(dj))
        


    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    Ystar = np.zeros((Y.shape[0], num_classes), dtype=int)
    n = Y.shape[0]
    for i in range(n):
        Ystar[i, Y[i]] = 1
    return Ystar


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                print(f'lidx={layer_idx}, i={i}, j={j}')
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
