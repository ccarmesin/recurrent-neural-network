import * as tf from '@tensorflow/tfjs';

let weights_ih, weights_ho;

// Learning rate
const lr = .1;

function build(inputNodes, hiddenNodes, outputNodes) {

    weights_ih = tf.randomNormal([hiddenNodes, inputNodes]);
    weights_ho = tf.randomNormal([outputNodes, hiddenNodes]);

}

build(2, 4, 1);
train(tf.tensor2d([0, 1], [2, 1]), tf.tensor2d([1], [1, 1]));


function feedForward(xs) {

    const hiddenOutputs = tf.matMul(weights_ih, xs).sigmoid();
    const prediction = tf.matMul(weights_ho, hiddenOutputs).sigmoid();
    return {
        hiddenOutputs: hiddenOutputs,
        prediction: prediction
    }

}

function backpropagation(inputs, ys, outputs) {

    // Outputs of the hidden layer
    const hiddenOutputs = outputs.hiddenOutputs;

    // Outputs of the last layer(output layer)
    const predictions = outputs.prediction;

    const outputError = predictions.sub(ys);
    const weights_ho_T = weights_ho.transpose();
    const hiddenErrors = tf.matMul(weights_ho_T, outputError);
    
    // Calculate the gradient of the output layer
    let gradientOutput = sigmoidDerivative(predictions);
    gradientOutput.mul(outputError);
    gradientOutput.mul(lr);
    
    // Reshape from [4] to [1, 4]
    gradientOutput = gradientOutput.reshape([1, -1]);

    // Change in weights from HIDDEN --> OUTPUT
    const deltaW_output = tf.matMul(hiddenOutputs, gradientOutput);
    weights_ho.add(deltaW_output);

    // Gradients for next layer, more back propagation!

    // Calculate the gradient of the hidden layer
    const gradient_hidden = sigmoidDerivative(hiddenOutputs);

    // Weight by errors and learning rate
    gradient_hidden.mul(hiddenErrors);
    gradient_hidden.mul(lr);

    // Change in weights from INPUT --> HIDDEN
    const inputs_T = inputs.transpose();
    const deltaW_hidden = tf.matMul(gradient_hidden, inputs_T);
    weights_ih.add(deltaW_hidden);



}

function train(xs, ys) {

    const outputs = feedForward(xs);
    return backpropagation(xs, ys, outputs);

}

function sigmoidDerivative(errors) {

    const gradientErrors = errors.dataSync().map(x => x * (1 - x));
    return tf.tensor(gradientErrors, errors.shape);

}

function crossEntropyLoss(yLabel, yPred) {

    const delta = Math.abs(yLabel - yPred);
    const firstPart = yLabel * Math.log(delta);
    const secondPart = (1 - yLabel) * Math.log(Math.abs(1 - delta));
    return firstPart - secondPart;

}
