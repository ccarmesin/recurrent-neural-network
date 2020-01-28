import * as tf from '@tensorflow/tfjs-core';

/**
 * Cross entropy loss
 *
 * @param {tensor} yPred
 * @param {tensor} yLabel
 * 
 * @return {tensor} cross entropy loss
 */
export function crossEntropy(yPred, yLabel) {

    return tf.sub(yPred, yLabel);

}

/**
 * Mean squared error
 *
 * @param {tensor} yPred
 * @param {tensor} yLabel
 * 
 * @return {tensor} mean squared error
 */
export function meanSquared(yPred, yLabel) {

    return predictions.sub(labels).square().mean();

}
