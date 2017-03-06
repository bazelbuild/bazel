google.charts.load('current', {packages: ['corechart', 'controls', 'table']});
google.charts.setOnLoadCallback(drawAllChart);

/**
 * Get data from /data/*.json and draw charts from it.
 */
function drawAllChart() {
  $.get('file_list?v=' + Date.now(), function(data) {
    /** @type {!Array<Dashboard>} */
    const dashboard = [];
    /** @type {!Array<ControlWrapper>} */
    const control = [];
    /** @type {!Array<LineChart>} */
    const chart = [];
    const options = [];
    /** @type {!Array<DataTable>} */
    const tableData = [];
    /** @type {!Array<Column>} */
    const columns = [];
    /** @type {!boolean} */
    let chartInit = false;
    /** @type {!number} */
    let targetNum = 0;

    /** @type {!Array<string>} */
    const filenames = data.trim().split('\n');

    // Make sure the length of the deffered object array is always > 1
    $.when.apply($, [0].concat(filenames.map(function(filename) {
      return $.getJSON("data/" + filename);
    }))).then(function(){
      /** @type {!Array<Object>} */
      let responses = [].slice.call(arguments, 1);
      for (let response of responses) {
        let data = response[0];

        if (!chartInit) {
          targetNum = data.buildTargetResults.length;
          initChartData(data.buildTargetResults, dashboard, control, chart, tableData, options);
          chartInit = true;
        }

        // Add rows for chart (including data)
        for (let i = 0; i < targetNum; ++i) {
          addRowsFromData(tableData[i], data.buildTargetResults[i].buildEnvResults);
        }
      }
      afterChartData(targetNum, dashboard, control, chart, columns, tableData, options);
    });
  });
}

/**
 * Initialize all the chart data (columns, options, divs and chart objects)
 * @param {!Array<Object>} buildTargetResults results for all build targets
 * @param {!Array<Dashboard>} dashboard all dashboards
 * @param {!Array<Control>} control all controls
 * @param {!Array<LineChart>} chart all charts
 * @param {!Array<DataTable>} tableData data for all charts
 * @param {!Array<Object>} options options for all charts
 */
function initChartData (buildTargetResults, dashboard, control, chart, tableData, options) {
  for (let i = 0; i < buildTargetResults.length; ++i) {
    const buildEnvResults = buildTargetResults[i].buildEnvResults;

    // add divs to #content
    $('<div id="target' + i + '" style="width: 100%; height: 600px"></div>')
        .appendTo('#content');
    $('<div id="control' + i + '" style="width: 100%; height: 100px"></div>')
        .appendTo('#content');

    // Dashboard
    dashboard[i] = new google.visualization.Dashboard(
      document.getElementById('target' + i));

    // Control
    control[i] = new google.visualization.ControlWrapper({
      'controlType': 'ChartRangeFilter',
      'containerId': 'control' + i,
      'options': {
        // Filter by the date axis.
        'filterColumnIndex': 1,
        'ui': {
          'chartType': 'LineChart',
          'chartOptions': {
            'chartArea': {'width': '70%'},
            'hAxis': {'baselineColor': 'none'}
          },
          'chartView': {
            'columns': [0, 2, 6]
          }
        }
      }
    });

    // Options for each chart (including title)
    options[i] = {
      title: buildTargetResults[i].buildTargetConfig.description,
      vAxis: { title: 'Elapsed time (s)' },
      hAxis: { title: 'Changes with pushed time' },
      tooltip: { isHtml: true, trigger: 'both' },
      intervals: { style: 'bars' },
      chartArea: {  width: '70%' }
    };

    // Create data table & add columns(line options)
    tableData[i] = new google.visualization.DataTable();
    addColumnsFromBuildEnv(tableData[i], buildEnvResults);

    // Create chart objects
    chart[i] = new google.visualization.ChartWrapper({
      'chartType': 'LineChart',
      'containerId': 'target' + i,
      'options': options[i],
      'view': { columns: [0, 2, 3, 4, 5, 6, 7, 8, 9] }
    });
  }
}

/**
 * Called after getting and filling chart data, draw all charts
 * @param {!number} targetNum number of target configs (charts)
 * @param {!Array<Dashboard>} dashboard all dashboards
 * @param {!Array<Control>} control all controls
 * @param {!Array<LineChart>} chart all charts
 * @param {!Array<Column>} columns columns of all charts
 * @param {!Array<DataTable>} tableData data for all charts
 * @param {!Array<Object>} options options for all charts
 */
function afterChartData (targetNum, dashboard, control, chart, columns, tableData, options) {
  // final steps to draw charts
  for (let i = 0; i < targetNum; ++i) {
    dashboard[i].bind(control[i], chart[i]);
    dashboard[i].draw(tableData[i]);

    // event
    columns[i] = [];
    for (let j = 0; j < tableData[i].getNumberOfColumns(); j++) {
      columns[i].push(j);
    }

    google.visualization.events.addListener(
        chart[i], 'select', (function (x) {
          return function () {
            hideOrShow(dashboard[x], chart[x], columns[x], tableData[x], options[x]);
          };
        })(i));
  }
}

/**
 * Add columns for each buildEnvResults/line.
 * @param {!LineChart} lineChart
 * @param {!Array<Object>} buildEnvResults build results
 */
function addColumnsFromBuildEnv (lineChart, buildEnvResults) {
  // Using datetime value as hAxis label makes intervals different,
  // so we use number instead.
  lineChart.addColumn('string', 'label index');
  lineChart.addColumn('number', 'numeric index');
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
  const rowNum = lineChart.getNumberOfRows();
  for (let j = 0; j < buildEnvResults[0].results.length; ++j) {
    const row = [buildEnvResults[0].results[j].datetime, rowNum + j];

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
 * @param {!string} codeVersion current code version
 * @return {!string} the html content
 */
function createCustomHTMLContent(arr, codeVersion) {
  let str = '<div style="padding:10px 10px 10px 10px;">';
  for (let i = 0; i < arr.length; ++i) {
    str += (i+1) + '-th run: ' + arr[i] + '<br>';
  }
  str += '<a href="https://github.com/bazelbuild/bazel/commit/' + codeVersion
      + '">commit</a></div>';
  return str;
}

/**
 * Hide or show one column/line in a chart.
 * @param {!Dashboard} dashboard the dashboard to operate
 * @param {!LineChart} chart the chart to operate
 * @param {!Column} columns columns of current chart
 * @param {!DataTable} tableData data for current chart
 * @param {!Object} options options for current chart
 */
function hideOrShow(dashboard, chart, columns, tableData, options) {
  const sel = chart.getChart().getSelection();
  // If selection length is 0, we deselected an element
  if (sel.length <= 0 || sel[0].row !== null) {
    return;
  }

  // Since real col[1] is hidden (numeric index),
  // the col that we are looking for should +1
  const col = sel[0].column + 1;
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
  dashboard.draw(view);
}
