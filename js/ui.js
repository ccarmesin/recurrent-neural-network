import {
    GoogleCharts
} from 'google-charts';

const lossTxt = document.getElementById('lossTxt'),
    accTxt = document.getElementById('accTxt'),
    lossGraph = document.getElementById('lossGraph'),
    accGraph = document.getElementById('accGraph');

let lossArr = [],
    accArr = [];

//Load the charts library with a callback
GoogleCharts.load(drawChart);

/**
 *
 * Draw some data to a given google chart
 *
 * @param {number} dataArr containing data to draw
 * @param {tensor} graph to draw to
 */
function drawChart(dataArr, graph) {

    const lineChart = new GoogleCharts.api.visualization.LineChart(graph);

    const options = {
        legend: 'none'
    }

    // Standard google charts functionality is available as GoogleCharts.api after load
    const data = GoogleCharts.api.visualization.arrayToDataTable(dataArr);

    lineChart.draw(data, options);

}

/**
 *
 * Log loss to lossTxt and draw in lossGraph
 *
 * @param {number} epoch gives the currentEpoch
 * @param {tensor} loss of the epoch
 */
export function logLoss(epoch, loss) {
    lossTxt.value += 'Epoch: ' + epoch + ' Loss: ' + loss + '\n';

    if (lossArr[0] === undefined) {
        lossArr.push(['epoch', 'loss']);
    }

    lossArr.push([epoch, loss]);

    GoogleCharts.load(() => drawChart(lossArr, lossGraph));
}

/**
 *
 * Log accuracy to accTxt and draw in accGraph
 *
 * @param {number} epoch gives the currentEpoch
 * @param {tensor} acc of the epoch
 */
export function logAccuracy(epoch, acc) {
    accTxt.value += 'Epoch: ' + epoch + ' Acc: ' + acc + '\n';

    if (accArr[0] === undefined) {
        accArr.push(['epoch', 'accuracy']);
    }

    accArr.push([epoch, acc]);

    GoogleCharts.load(() => drawChart(accArr, accGraph));
}
