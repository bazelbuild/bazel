google.charts.load('current', {packages: ['corechart']});
google.charts.setOnLoadCallback(drawAllChart);

/**
 * Get data from /data.json and draw charts from it.
 */
function drawAllChart() {
  $.getJSON('data.json', function(data) {
    /** @type {!Array<LineChart>} */
    const chart = [];
    const options = [];
    /** @type {!Array<DataTable>} */
    const tableData = [];
    /** @type {!Array<Column>} */
    const columns = [];
    const buildTargetResultsLen = data.buildTargetResults.length;

    // add divs to #content
    for (let i = 0; i < buildTargetResultsLen; ++i) {
      $('<div id="target' + i + '" style="width: 100%; height: 600px"></div>')
          .appendTo('#content');
    }

    // draw chart for every BuildTargetResult
    for (let i = 0; i < buildTargetResultsLen; ++i) {
      const buildEnvResults = data.buildTargetResults[i].buildEnvResults;

      // Add columns and rows for chart
      tableData[i] = new google.visualization.DataTable();
      addColumnsFromBuildEnv(tableData[i], buildEnvResults);
      addRowsFromData(tableData[i], buildEnvResults);

      options[i] = {
        title: data.buildTargetResults[i].buildTargetConfig.description,
        tooltip: { isHtml: true, trigger: 'both'},
        intervals: { style: 'bars' },
        chartArea: {  width: '70%' }
      };

      chart[i] = new google.visualization.LineChart(
          document.getElementById('target' + i));
      chart[i].draw(tableData[i], options[i]);

      // ---------------------------------- event

      columns[i] = [];
      for (let j = 0; j < tableData[i].getNumberOfColumns(); j++) {
        columns[i].push(j);
      }

      google.visualization.events.addListener(
          chart[i], 'select', (function (x) {
            return function () {
              hideOrShow(chart[x], columns[x], tableData[x], options[x]);
            };
          })(i));
    }
  });
}

/**
 * Add columns for each buildEnvResults/line.
 * @param {!LineChart} lineChart
 * @param {!Array<Object>} buildEnvResults build results
 */
function addColumnsFromBuildEnv (lineChart, buildEnvResults) {
  lineChart.addColumn('string', 'version');
  for (let buildEnvResult of buildEnvResults) {
    lineChart.addColumn(
        'number', buildEnvResult.config.description);
    lineChart.addColumn({type:'number', role:'interval'});
    lineChart.addColumn({type:'number', role:'interval'});
    lineChart.addColumn(
        {'type': 'string', 'role': 'tooltip', 'p': {'html': true}});
  }
}

/**
 * Add rows for each code version.
 * @param {!LineChart} lineChart
 * @param {!Array<Object>} buildEnvResults build results
 */
function addRowsFromData (lineChart, buildEnvResults) {
  for (let j = 0; j < buildEnvResults[0].results.length; ++j) {
    const row = [buildEnvResults[0].results[j].codeVersion.substr(0, 10)];
    for (let buildEnvResult of buildEnvResults) {
      const singleBuildResult = buildEnvResult.results[j];

      const ave = getAverage(singleBuildResult.results);
      const sd = getStandardDeviation(singleBuildResult.results, ave);
      row.push(ave);
      row.push(ave - sd);
      row.push(ave + sd);
      row.push(
          createCustomHTMLContent(
              singleBuildResult.results, singleBuildResult.codeVersion));
    }
    lineChart.addRow(row);
  }
}

/**
 * Get average of an array.
 * @param {!Array<number>} arr
 * @return {!number} the average
 */
function getAverage(arr) {
  let ave = arr.reduce(function(a, b) { return a + b; });
  ave /= arr.length;
  return ave;
}

/**
 * Get standard deviation of an array.
 * @param {!Array<number>} arr
 * @param {!number} ave average of the array
 * @return {!number} the standard deviation
 */
function getStandardDeviation(arr, ave) {
  let sd = 0;
  for (let item of arr) {
    const diff = ave - item;
    sd += diff * diff;
  }
  sd = Math.sqrt(sd);
  return sd;
}

/**
 * Create html content as tooltip.
 * @param {!Array<number>} arr array of build results
 * @return {!string} the html content
 */
function createCustomHTMLContent(arr) {
  let str = '<div style="padding:10px 10px 10px 10px;">';
  for (let i = 0; i < arr.length; ++i) {
    str += (i+1) + '-th run: ' + arr[i] + '<br>';
  }
  str += '</div>';
  return str;
}

/**
 * Hide or show one column/line in a chart.
 * @param {!LineChart} chart the chart to operate
 * @param {!Column} columns columns of current chart
 * @param {!DataTable} tableData data for current chart
 * @param {!Object} options options for current chart
 */
function hideOrShow(chart, columns, tableData, options) {
  const sel = chart.getSelection();
  // If selection length is 0, we deselected an element
  if (sel.length <= 0 || sel[0].row !== null) {
    return;
  }

  const col = sel[0].column;
  if (columns[col] == col) {
    // Hide the data series
    columns[col] = {
      label: tableData.getColumnLabel(col),
      type: tableData.getColumnType(col),
      calc: function () {
          return null;
      }
    };
  } else {
    // Show the data series
    columns[col] = col;
  }
  const view = new google.visualization.DataView(tableData);
  view.setColumns(columns);
  chart.draw(view, options);
}