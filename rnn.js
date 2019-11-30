import * as tf from '@tensorflow/tfjs';

let weights_ih, weights_ho, weights_hs, hiddenStates, predictions, losses, xs, ys;

// Learning rate
const lr = .1, FORWARD = "forward", BACKWARD = "backward";

function build(inputNodes, hiddenNodes, outputNodes) {

    // Weight matrix from input to hidden
    weights_ih = tf.randomNormal([hiddenNodes, inputNodes]);

    // Weight matrix of the hidden state
    weights_hs = tf.randomNormal([hiddenNodes, hiddenNodes]);

    // Weight matrix from hidden to output
    weights_ho = tf.randomNormal([outputNodes, hiddenNodes]);

    // Init hiddenStates log to store all hiddenStates for backprop
    hiddenStates = [];

    // Init predictions log to store all preds for backprop
    predictions = [];

    // Init losses log to store all losses for backprop
    losses = [];

}

build(4, 4, 4);
prepareData();
train(0, FORWARD);

function prepareData() {

    // Word "hello" encoded in one hot
    xs = tf.tensor2d([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]);

    // Word "elloh" encoded in one hot
    ys = tf.tensor2d([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]);

    // Init hidden state log with zeros because the first sequence has no hiddenState
    hiddenStates[0] = tf.zeros([4, 1]);

}

// xs = input (onehot endcoded); ys = output (onehot encoded); currentChar = current sequence the feed into the network
function feedForward(xs, ys, currentChar) {

    // Multiply weight matrix of the hiddenState with the hiddenState of previous timestep
    const hiddenStateOutput = tf.matMul(weights_hs, hiddenStates[currentChar]);

    // Multiply weight matrix from input to hidden with the input data, like in a simple neural network
    const inputOutput = tf.matMul(weights_ih, xs);

    // Marry history and input of current timestep and activate them
    hiddenStates[currentChar + 1] = tf.add(hiddenStateOutput, inputOutput).sigmoid();

    // The output of the outputLayer is the prediction of the network
    predictions[currentChar] = tf.matMul(weights_ho, hiddenStates[currentChar + 1]).sigmoid();

    // Calculate the loss
    losses[currentChar] = tf.losses.sigmoidCrossEntropy(ys, predictions[currentChar]);

    
    return train(currentChar + 1, FORWARD);

}

function train(currentChar, pass) {

    // Encode current char
    const wordLength = 4;
    const inputCharEncoding = xs.slice([currentChar, 0], [1]).reshape([wordLength, 1]);
    const outputCharEncoding = ys.slice([currentChar, 0], [1]).reshape([wordLength, 1]);

    // Change to backwardPass
    if (currentChar === wordLength) {
        pass = BACKWARD;
        currentChar--;
    }
    
    // End training
    if(currentChar === -1) {
        console.log("Training end");
        return;
    }

    if (pass === FORWARD) {
        // FeedForward
        console.log("Forward: " + currentChar);
        return feedForward(inputCharEncoding, outputCharEncoding, currentChar);
    } else {
        // BackwardPass
        console.log("Backward: " + currentChar);
        return backprop(inputCharEncoding, outputCharEncoding, currentChar)
    }

}

function backprop(xs, ys, currentChar) {

    const outputError = predictions[currentChar].sub(ys);
    const weights_ho_T = weights_ho.transpose();
    const hiddenErrors = tf.matMul(weights_ho_T, outputError);

    // Calculate the gradient of the output layer
    let gradientOutput = sigmoidDerivative(predictions[currentChar]);
    gradientOutput.mul(outputError);
    gradientOutput.mul(lr);

    // Reshape from [4] to [1, 4]
    gradientOutput = gradientOutput.reshape([1, -1]);

    // Change in weights from HIDDEN --> OUTPUT
    const deltaW_output = tf.matMul(hiddenStates[currentChar + 1], gradientOutput);
    weights_ho.add(deltaW_output);

    // Gradients for next layer, more back propagation!

    // Calculate the gradient of the hidden layer
    const gradient_hidden = sigmoidDerivative(hiddenStates[currentChar + 1]);

    // Weight by errors and learning rate
    gradient_hidden.mul(hiddenErrors);
    gradient_hidden.mul(lr);

    // Change in weights from INPUT --> HIDDEN
    const xs_T = xs.transpose();
    const deltaW_hidden = tf.matMul(gradient_hidden, xs_T);
    weights_ih.add(deltaW_hidden);

    
    return train(currentChar - 1, BACKWARD);

}

function sigmoidDerivative(errors) {

    const gradientErrors = errors.dataSync().map(x => x * (1 - x));
    return tf.tensor(gradientErrors, errors.shape);

}

/*
// currentChar = char in sequence; wordLength = length of the batch to feed in
function feedForward(currentChar, wordLength) {

    // Encode current char
    const inputCharEncoding = xs.slice([currentChar, 0], [1]).reshape([wordLength, 1]);
    const outputCharEncoding = ys.slice([currentChar, 0], [1]).reshape([wordLength, 1]);

    // Multiply weight matrix of the hiddenState with the hiddenState of previous timestep
    const hiddenStateOutput = tf.matMul(weights_hs, hiddenState);

    // Multiply weight matrix from input to hidden with the input data, like in a simple neural network
    const inputOutput = tf.matMul(weights_ih, inputCharEncoding);

    // Marry history and input of current timestep and activate them
    hiddenState = tf.add(hiddenStateOutput, inputOutput).sigmoid();

    // The output of the outputLayer is the prediction of the network
    const prediction = tf.matMul(weights_ho, hiddenState).sigmoid();

    // Calculate the loss
    const loss = tf.losses.sigmoidCrossEntropy(outputCharEncoding, prediction);

    prediction.print();
    loss.print();
    console.log("----");



    return feedForward(currentChar + 1, wordLength);

}*/
