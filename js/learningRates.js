import * as tf from '@tensorflow/tfjs';

/**
 * Just returning the given learning Rate
 *
 * lr: learn rate to apply
 */
export function fixedLr(lr) {

    return lr;

}

/**
 * Calculate lr as part of the maximum loss the network made during training
 *
 * currentEpoch:
 * totalEpochs:
 */
export function dynamicLr(currentEpoch, totalEpochs) {
    
    return Math.exp(-(currentEpoch / totalEpochs)) - .35

}

/**
 * Increase and decrease the learningRate multiple times during training to avoid local minima
 *
 * sequence: how many epochs the learningRate will stay contstant
 * max: maximum learningRate we want to apply
 * min: minimum learningRate we want to apply
 * dev: deviation from normal distribution
 */
export function hybridLr(sequence, max, min, dev) {

    // Create a random binary number to estimate which learningRate we should apply next(0 = low mean, 1 = high mean)
    const randomBinary = Math.round(Math.random());
    let lr = 0;
    if(randomBinary === 1) {
        
        // Use max mean
        lr = tf.randomNormal([1, 1], max, dev);
        
    } else {
        
        // Use min mean
        lr = tf.randomNormal([1, 1], min, dev);
        
    }
    
    lr = lr.dataSync()[0];
    return Math.abs(lr);
    
}