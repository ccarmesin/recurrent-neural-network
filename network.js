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

function backpropagation(ys, outputs) {

    // Outputs of the hidden layer
    const hiddenOutputs = outputs.hiddenOutputs;

    // Outputs of the last layer(output layer)
    const predictions = outputs.prediction;

    const outputError = ys.sub(predictions);
    const weights_ho_T = weights_ho.transpose();
    const hiddenErrors = tf.matMul(weights_ho_T, outputError);
    const gradientOutput = sigmoidDerivative(hiddenErrors);

    gradientOutput.mul(outputError);
    gradientOutput.mul(lr);

    // Change in weights from HIDDEN --> OUTPUT
    gradientOutput.print();
    hiddenOutputs.print();
    const deltaW_output = tf.matMul(gradientOutput, hiddenOutputs);
    weights_ho.add(deltaW_output);



}

function train(xs, ys) {

    const outputs = feedForward(xs);
    return backpropagation(ys, outputs);

}

function sigmoidDerivative(errors) {

    const gradientErrors = errors.dataSync().map(x => x * (1 - x));
    return tf.tensor(gradientErrors);

}
