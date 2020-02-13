import * as losses from './losses';
import * as datasets from './datasets';
import * as ui from './ui';
import {
    Rnn
} from './rnn';

const architecture = {
    inputNodes: 4,
    hiddenNodes: 4,
    outputNodes: 4
}

const hyperparameters = {
    loss: losses.crossEntropy,
    lr: .06,
    epochs: 10
}

const uiCallbacks = {
    logLoss: ui.logLoss,
    logAccuracy: ui.logAccuracy
}

const rnn = new Rnn(architecture, hyperparameters, datasets.hello(), uiCallbacks);
rnn.train();
/*rnn.predict(0).print();
rnn.predict(1).print();
rnn.predict(2).print();
rnn.predict(3).print();
rnn.predict(4).print();*/