const lossTxt = document.getElementById('lossTxt'),
    accTxt = document.getElementById('accTxt'),
    lossGraph = document.getElementById('lossGraph'),
    accGraph = document.getElementById('accGraph');

let lossArr = [],
    accArr = [];

import {
    GoogleCharts
} from 'google-charts';

//Load the charts library with a callback
GoogleCharts.load(drawChart);

function drawChart(dataArr, graph) {

    const lineChart = new GoogleCharts.api.visualization.LineChart(graph);

    const options = {
        legend: 'none'
    }

    // Standard google charts functionality is available as GoogleCharts.api after load
    const data = GoogleCharts.api.visualization.arrayToDataTable(dataArr);

    lineChart.draw(data, options);

}

export function logLoss(epoch, loss) {
    lossTxt.value += 'Epoch: ' + epoch + ' Loss: ' + loss + '\n';

    if (lossArr[0] === undefined) {
        lossArr.push(['epoch', 'loss']);
    }
    
    lossArr.push([epoch, loss]);

    GoogleCharts.load(() => drawChart(lossArr, lossGraph));
}

export function logAccuracy(epoch, acc) {
    accTxt.value += 'Epoch: ' + epoch + ' Acc: ' + acc + '\n';

    if (accArr[0] === undefined) {
        accArr.push(['epoch', 'accuracy']);
    }
    
    accArr.push([epoch, acc]);

    GoogleCharts.load(() => drawChart(accArr, accGraph));
}
