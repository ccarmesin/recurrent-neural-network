import * as tf from '@tensorflow/tfjs';

export class Rnn {

    constructor(architecture, hyperparameters, dataset, uiCallbacks) {

        // Hyperparameters
        // Learningrate is calculated by the learnFunction during training
        this.learnFunction = hyperparameters.lr;
        this.lr = .01;

        // Hyperparameters
        this.epochs = hyperparameters.epochs;

        // Dataset to train the model on
        this.dataset = dataset;

        // Optimizer(e.g. cross-entropy-loss)
        this.loss = hyperparameters.loss;

        // Weight matrix from input to hidden
        this.weights_ih = tf.randomNormal([architecture.hiddenNodes, architecture.inputNodes]);

        // Weight matrix of the hidden state
        this.weights_hh = tf.randomNormal([architecture.hiddenNodes, architecture.hiddenNodes]);

        // Weight matrix from hidden to output
        this.weights_ho = tf.randomNormal([architecture.outputNodes, architecture.hiddenNodes]);

        // Don't know yet while this is working
        this.dhnext = tf.zeros([dataset.classes, 1]);

        // Init hiddenStates log to store all hiddenStates for backprop
        this.hiddenStates = [];

        // Init hidden state log with zeros because the first sequence has no hiddenState
        this.hiddenStates[-1] = tf.zeros([dataset.classes, 1]);

        // Init predictions log to store all preds for backprop
        this.predictions = [];

        // Store input as first prediction, so that the input to the input layer is the general input of the network
        this.predictions[-1] = dataset.xs[0];

        // Set callback for updating the UI during training
        this.ui = uiCallbacks;

        // States
        this.FORWARD = "forward";
        this.BACKWARD = "backward";

    }

    /**
     * Train the model
     */
    async train() {

        for (let epoch = 0; epoch < this.epochs; epoch++) {

            this.execute(0, this.FORWARD);
            this.calculateLearnState(epoch);

        }

    }

    /**
     * Helper function for the training process
     *
     * @param {number} t indicator for the sequence of the network(current timestep)
     * @param {number} pass contains information about the current pass(forward or backward)
     *
     * @return {function} feedforward
     * @return {function} backprop
     * @return {function} null if epoch training is complete
     */
    async execute(t, pass) {

        // Change to backwardPass
        if (t === this.dataset.timesteps) {
            pass = this.BACKWARD;
            t--;
        }

        // End training
        if (t === -1) {
            //console.log("Training end");
            return;
        }

        if (pass === this.FORWARD) {
            // FeedForward
            //console.log("Forward: " + t);
            return this.feedForward(t);
        } else {
            // BackwardPass
            //console.log("Backward: " + t);
            return this.backprop(t);
        }

    }

    /**
     * Pass data though the network
     *
     * @param {number} t indicator for the sequence of the network(current timestep)
     *
     * @return {function} execute the next sequence
     */
    feedForward(t) {

        // Multiply weight matrix from input to hidden with the input data, like in a simple neural network
        const inputOutput = tf.matMul(this.weights_ih, this.dataset.xs[t]);

        // Multiply weight matrix of the hiddenState with the hiddenState of previous timestep
        const hiddenStateOutput = tf.matMul(this.weights_hh, this.hiddenStates[t - 1]);

        // Marry history and input of current timestep
        this.hiddenStates[t] = tf.add(hiddenStateOutput, inputOutput).tanh();

        // The output of the outputLayer is the prediction of the network
        this.predictions[t] = tf.matMul(this.weights_ho, this.hiddenStates[t]).transpose().softmax().transpose();

        return this.execute(t + 1, this.FORWARD);

    }

    /**
     * Backpropagte though the network
     *
     * @param {number} t indicator for the sequence of the network(current timestep)
     *
     * @return {function} execute the next sequence
     */
    backprop(t) {

        // Calculate error of output layer
        const outputError = this.loss(this.predictions[t], this.dataset.ys[t]);

        // Calculate the gradient of the output layer
        let output_derivative = this.sigmoidDerivative(this.predictions[t]);
        output_derivative = output_derivative.mul(outputError);
        output_derivative = output_derivative.mul(this.lr);

        // Change in weights from HIDDEN --> OUTPUT
        const deltaW_output = tf.matMul(output_derivative, this.predictions[t - 1].transpose());
        this.weights_ho = tf.sub(this.weights_ho, deltaW_output);

        // Gradients for next layer, more back propagation!

        // Calculate hiddenError: (weights_HO_T x outputErrors) * dhnext
        let hiddenErrors = tf.matMul(this.weights_ho.transpose(), outputError);
        hiddenErrors = tf.add(hiddenErrors, this.dhnext);

        // Calculate the gradient of the hidden layer(a little bit different than in a normal NN)
        let hidden_derivative = tf.sub(1, this.hiddenStates[t].square());

        // Weight by errors and learning rate
        hidden_derivative = hidden_derivative.mul(hiddenErrors);
        hidden_derivative = hidden_derivative.mul(this.lr);

        // Change in weights from INPUT --> HIDDEN
        const deltaW_ih = tf.matMul(hidden_derivative, this.dataset.xs[t].transpose());
        this.weights_ih = this.weights_ih.sub(deltaW_ih);

        // Change weights from HIDDEN --> HIDDEN
        const deltaW_hh = tf.matMul(hidden_derivative, this.hiddenStates[t - 1].transpose());
        this.weights_hh = this.weights_hh.sub(deltaW_hh);

        this.dhnext = tf.matMul(this.weights_hh, hidden_derivative);

        return this.execute(t - 1, this.BACKWARD);

    }

    /**
     * Compute the derivative of the sigmoid activation function
     *
     * @param {tensor} errors raw
     *
     * @return {tensor} errors activated
     */
    sigmoidDerivative(errors) {

        const gradientErrors = errors.dataSync().map(x => x * (1 - x));
        return tf.tensor(gradientErrors, errors.shape);

    }

    /**
     * Calculate the current state of the network(including lr, loss and accuracy)
     *
     * @param {number} epoch of the network
     */
    async calculateLearnState(epoch) {

        // Loss for the entire sequence
        let sequenceLoss = 0;
        let sequenceAccuracy = 0;

        for (let t = 0; t < this.dataset.timesteps; t++) {

            // Loss at this timestep
            const currentLoss = tf.losses.sigmoidCrossEntropy(this.dataset.ys[t].transpose(), this.predictions[t].transpose()).dataSync()[0];
            sequenceLoss += currentLoss;
            const predIndex = this.predictions[t].argMax().dataSync()[0];
            const lblIndex = this.dataset.ys[t].argMax().dataSync()[0];
            if (predIndex === lblIndex) sequenceAccuracy++;

        }

        // Estimate real acc by dividing the timesteps
        sequenceAccuracy = sequenceAccuracy / this.dataset.timesteps;

        // Calculate learnRate for the next sequence based on given learnFunction
        this.lr = this.learnFunction(epoch, this.epochs);


        this.ui.logLoss(epoch, sequenceLoss);
        this.ui.logAccuracy(epoch, sequenceAccuracy);
        console.log('Epoch: ', epoch, 'Loss: ', sequenceLoss, 'Acc: ', sequenceAccuracy);

    }

    /**
     * Make predictions on the model
     * Make sure you have train it before:)
     *
     * @param {number} t gives the timesequence to make a prediction on
     * 
     * @return {tensor} predictions of the model
     * 
     */
    predict(t) {

        // Multiply weight matrix from input to hidden with the input data, like in a simple neural network
        const inputOutputRaw = tf.matMul(this.weights_ih, this.dataset.xs[t]);

        // Multiply weight matrix of the hiddenState with the hiddenState of previous timestep
        const hiddenStateOutputRaw = tf.matMul(this.weights_hh, this.hiddenStates[t - 1]);

        const hiddenOutput = tf.add(hiddenStateOutputRaw, inputOutputRaw).tanh();

        return tf.matMul(this.weights_ho, hiddenOutput).transpose().sigmoid().transpose();

    }

}
