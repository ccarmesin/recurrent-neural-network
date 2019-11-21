import * as tf from '@tensorflow/tfjs';

const xTrain = [1, 5, 3, 4];
const yTrain = [2, 4, 6, 8];
const weight = 2;

let y = 0,
    yTanh = 0;

for (let i = 0; i < xTrain.length; i++) {

    let prediction = y * weight;
    let loss = crossEntropyLoss(yTrain[i], prediction);
    console.log(loss);
    console.log("----");

}

function crossEntropyLoss(yLabel, yPred) {
    
    const delta = Math.abs(yLabel - yPred);
    const firstPart = yLabel * Math.log(delta);
    console.log(firstPart);
    const secondPart = (1 - yLabel) * Math.log(Math.abs(1 - delta));
    console.log(secondPart);
    return firstPart - secondPart;
    
}