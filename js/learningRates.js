import * as tf from '@tensorflow/tfjs';

/**
 * Just returning the given learning Rate
 *
 * @param {number} lr
 *
 * @return {number} calculated lr
 */
export function fixedLr(lr) {

    return lr;

}

/**
 * Calculate lr as part of the maximum loss the network made during training
 *
 * @param {number} currentEpoch
 * @param {number} totalEpochs
 *
 * @return {number} calculated lr
 */
export function dynamicLr(currentEpoch, totalEpochs) {

    return Math.exp(-(currentEpoch / totalEpochs)) - .35

}

/**
 * Increase and decrease the learningRate multiple times during training to avoid local minima
 *
 * @param {number} sequence = how many epochs the learningRate will stay contstant
 * @param {number} max = maximum learningRate we want to apply
 * @param {number} min = minimum learningRate we want to apply
 * @param {number} dev = deviation from normal distribution
 *
 * @return {number} calculated lr
 */
export function hybridLr(sequence, max, min, dev) {

    // Create a random binary number to estimate which learningRate we should apply next(0 = low mean, 1 = high mean)
    const randomBinary = Math.round(Math.random());
    let lr = 0;
    if (randomBinary === 1) {

        // Use max mean
        lr = tf.randomNormal([1, 1], max, dev);

    } else {

        // Use min mean
        lr = tf.randomNormal([1, 1], min, dev);

    }

    lr = lr.dataSync()[0];
    return Math.abs(lr);

}