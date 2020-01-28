import * as tf from '@tensorflow/tfjs-core';

export function hello() {
    
    console.log();

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

    return formatDataset(xs, ys);

}

// xs and ys are tensors
function formatDataset(xs, ys) {

    let input = [];
    let label = [];
    
    // Get word length
    const wordLength = xs.shape[0];
    const alphabetLength = xs.shape[1];
    console.log("Word length is " + wordLength + " and alphabet contains " + alphabetLength + " chars.");

    for (let currentChar = 0; currentChar < wordLength; currentChar++) {

        const inputCharEncoding = xs.slice([currentChar], [1]).transpose();
        const outputCharEncoding = ys.slice([currentChar], [1]).transpose();

        input[currentChar] = inputCharEncoding;
        label[currentChar] = outputCharEncoding;

    }
    
    return {
        xs: input,
        ys: label,
        classes: alphabetLength,
        timesteps: wordLength
    }

}
