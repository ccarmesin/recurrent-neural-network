import * as tf from '@tensorflow/tfjs';

let weights_ih, weights_ho, bias_ih, bias_ho;

// Learning rate
const lr = .05;

function build(inputNodes, hiddenNodes, outputNodes) {

    // Weights
    weights_ih = tf.randomNormal([hiddenNodes, inputNodes]);
    weights_ho = tf.randomNormal([outputNodes, hiddenNodes]);

    // Bias values
    bias_ih = tf.randomUniform([1]).asScalar();
    bias_ho = tf.randomUniform([1]).asScalar();

}

build(2, 4, 1);
for (let i = 0; i < 1000; i++) {

    train(tf.tensor2d([0, 1], [2, 1]), tf.tensor2d([1], [1, 1]));

    const pred = feedForward(tf.tensor2d([0, 1], [2, 1]));
    pred.prediction.print();


}
const pred = feedForward(tf.tensor2d([0, 1], [2, 1]));
pred.prediction.print();

function feedForward(xs) {

    const hiddenOutputs = tf.matMul(weights_ih, xs).sigmoid();
    const prediction = tf.matMul(weights_ho, hiddenOutputs).sigmoid();
    return {
        inputs: xs,
        hiddenOutputs: hiddenOutputs,
        prediction: prediction
    }

}

function backpropagation(labels, outputs) {

    let output_error = tf.losses.sigmoidCrossEntropy(outputs.prediction, labels);
    output_error = output_error.reshape(labels.shape);
    let output_derivative = sigmoidDerivative(outputs.prediction);
    output_derivative = output_derivative.mul(output_error);
    output_derivative = output_derivative.mul(lr);

    weights_ho = weights_ho.add(output_derivative);

    const whoT = weights_ho.transpose();

    const hidden_error = tf.matMul(whoT, output_error);
    let hidden_derivative = sigmoidDerivative(outputs.hiddenOutputs);
    hidden_derivative = hidden_derivative.mul(hidden_error);
    hidden_derivative = hidden_derivative.mul(lr);

    weights_ih = weights_ih.add(hidden_derivative);

}

function train(xs, ys) {

    const outputs = feedForward(xs);
    return backpropagation(ys, outputs);

}

function sigmoidDerivative(errors) {

    const gradientErrors = errors.dataSync().map(x => x * (1 - x));
    return tf.tensor(gradientErrors, errors.shape);

}

function crossEntropyLoss(predictions, labels) {

    const delta = Math.abs(labels - predictions);
    const firstPart = labels * Math.log(delta);
    const secondPart = (1 - labels) * Math.log(Math.abs(1 - delta));
    return firstPart - secondPart;

}

function meanSquaredError(predictions, labels) {

    const cost = tf.sub(predictions, labels);
    return tf.mul(cost, cost);

}
