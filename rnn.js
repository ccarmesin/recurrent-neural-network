import * as tf from '@tensorflow/tfjs';

let weights_ih, weights_ho, weights_hh, hiddenStates, predictions, input, label, wordLength, alphabetLength, dhnext;

// Learning rate
const lr = .1,
    FORWARD = "forward",
    BACKWARD = "backward";

function build(inputNodes, hiddenNodes, outputNodes) {

    // Weight matrix from input to hidden
    weights_ih = tf.randomNormal([hiddenNodes, inputNodes]);

    // Weight matrix of the hidden state
    weights_hh = tf.randomNormal([hiddenNodes, hiddenNodes]);

    // Weight matrix from hidden to output
    weights_ho = tf.randomNormal([outputNodes, hiddenNodes]);

    // Init hiddenStates log to store all hiddenStates for backprop
    hiddenStates = [];

    // Init predictions log to store all preds for backprop
    predictions = [];

    // Init list of inputs to each timestep
    input = [];

    // Init list of labels to each timestep
    label = [];

}

function prepareData() {

    // Word "hello" encoded in one hot
    const xs = tf.tensor2d([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]);

    // Word "elloh" encoded in one hot
    const ys = tf.tensor2d([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]);

    // Get word length
    wordLength = xs.shape[0];
    alphabetLength = xs.shape[1];
    console.log("Word length is " + wordLength + " and alphabet contains " + alphabetLength + " chars.");

    for (let currentChar = 0; currentChar < wordLength; currentChar++) {

        const inputCharEncoding = xs.slice([currentChar], [1]).transpose();
        const outputCharEncoding = ys.slice([currentChar], [1]).transpose();

        input[currentChar] = inputCharEncoding;
        label[currentChar] = outputCharEncoding;

    }

    // Store input as first prediction, so that the input to the input layer is the general input of the network
    predictions[-1] = input[0];

    // Init hidden state log with zeros because the first sequence has no hiddenState
    hiddenStates[-1] = tf.zeros([4, 1]);

    // Don't know yet while this is working
    dhnext = tf.zeros([4, 1]);

}

execute();

function execute() {

    build(4, 4, 4);
    prepareData();
    for (let i = 0; i < 500; i++) {

        train(0, FORWARD);

    }

    predict(0).print();
    predict(1).print();
    predict(2).print();
    predict(3).print();
    predict(4).print();

}

function train(currentChar, pass) {

    // Change to backwardPass
    if (currentChar === wordLength) {
        pass = BACKWARD;
        currentChar--;
    }

    // End training
    if (currentChar === -1) {
        //console.log("Training end");
        return;
    }

    if (pass === FORWARD) {
        // FeedForward
        //console.log("Forward: " + currentChar);
        return feedForward(currentChar);
    } else {
        // BackwardPass
        //console.log("Backward: " + currentChar);
        return backprop(currentChar);
    }

}

// t = current sequence the feed into the network
function feedForward(t) {

    // Multiply weight matrix from input to hidden with the input data, like in a simple neural network
    const inputOutput = tf.matMul(weights_ih, input[t]);

    // Multiply weight matrix of the hiddenState with the hiddenState of previous timestep
    const hiddenStateOutput = tf.matMul(weights_hh, hiddenStates[t - 1]);

    // Marry history and input of current timestep
    hiddenStates[t] = tf.add(hiddenStateOutput, inputOutput).tanh();

    // The output of the outputLayer is the prediction of the network
    predictions[t] = tf.matMul(weights_ho, hiddenStates[t]).transpose().softmax().transpose();

    return train(t + 1, FORWARD);

}

// t = current timestep
function backprop(t) {

    // Calculate error of output layer
    const outputError = loss(predictions[t], label[t]);

    // Calculate the gradient of the output layer
    let output_derivative = sigmoidDerivative(predictions[t]);
    output_derivative = output_derivative.mul(outputError);
    output_derivative = output_derivative.mul(lr);

    // Change in weights from HIDDEN --> OUTPUT
    const deltaW_output = tf.matMul(output_derivative, predictions[t - 1].transpose());
    weights_ho = tf.sub(weights_ho, deltaW_output);

    // Gradients for next layer, more back propagation!

    // Calculate hiddenError: (weights_HO_T x outputErrors) * dhnext
    let hiddenErrors = tf.matMul(weights_ho.transpose(), outputError);
    hiddenErrors = tf.add(hiddenErrors, dhnext);

    // Calculate the gradient of the hidden layer(a little bit different than in a normal NN)
    let hidden_derivative = tf.sub(1, hiddenStates[t].square());

    // Weight by errors and learning rate
    hidden_derivative = hidden_derivative.mul(hiddenErrors);
    hidden_derivative = hidden_derivative.mul(lr);

    // Change in weights from INPUT --> HIDDEN
    const deltaW_ih = tf.matMul(hidden_derivative, input[t].transpose());
    weights_ih = weights_ih.sub(deltaW_ih);

    // Change weights from HIDDEN --> HIDDEN
    const deltaW_hh = tf.matMul(hidden_derivative, hiddenStates[t - 1].transpose());
    weights_hh = weights_hh.sub(deltaW_hh);

    dhnext = tf.matMul(weights_hh, hidden_derivative);

    return train(t - 1, BACKWARD);

}

// Compute the derivative of the sigmoid activation function
function sigmoidDerivative(errors) {

    const gradientErrors = errors.dataSync().map(x => x * (1 - x));
    return tf.tensor(gradientErrors, errors.shape);

}

// prediction = prediction of the network, label = what it has to be(the solution)
function loss(prediction, label) {

    return tf.sub(prediction, label);

}

// t = current sequence the feed into the network
function predict(t) {

    //console.log("Hidden state " + t);
    //hiddenStates[t].print();

    // Multiply weight matrix from input to hidden with the input data, like in a simple neural network
    const inputOutputRaw = tf.matMul(weights_ih, input[t]);

    // Multiply weight matrix of the hiddenState with the hiddenState of previous timestep
    const hiddenStateOutputRaw = tf.matMul(weights_hh, hiddenStates[t - 1]);

    const hiddenOutput = tf.add(hiddenStateOutputRaw, inputOutputRaw).tanh();

    return tf.matMul(weights_ho, hiddenOutput).sigmoid();

}
