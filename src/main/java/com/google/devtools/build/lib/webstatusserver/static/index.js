function showData() {
  renderTestList(getTestsData());
}

function getTestsData() {
  // TODO(bazel-team): change it to async callback retrieving data in background
  // (for simplicity this is synchronous now)
  return $.ajax({
      type: 'GET',
      url: document.URL + 'tests/list',
      async: false
  }).responseJSON;
}

function renderTestList(tests) {
  var rows = d3.select('#testsList')
    .selectAll()
    .data(tests)
    .enter().append('div')
    .classed('row', true);

  // target(s) name(s)
  rows.append('div').classed('cell', true).text(function(j) {
    if (j.targets.length == 1) {
      return j.targets[0];
    }
    if (j.targets.length == 0) {
      return 'Unknown target.';
    }
    return j.targets;
  });

  // start time
  rows.append('div').classed('cell', true).text(function(j) {
    // Pad value with 2 zeroes
    function pad(value) {
      return value < 10 ? '0' + value : value;
    }

    var
      date = new Date(j.startTime),
      today = new Date(Date.now()),
      h = pad(date.getHours()),
      m = pad(date.getMinutes()),
      dd = pad(date.getDay()),
      mm = pad(date.getMonth()),
      yy = date.getYear(),
      day;

    // don't show date if ran today
    if (dd != today.getDay() && mm != today.getMonth() &&
        yy != today.getYear()) {
      day = ' on ' + yy + '-' + mm + '-' + dd;
    } else {
      day = '';
    }
    return h + ':' + m;
  });

  // status
  rows.append('div').classed('cell', true).text(function(j) {
    return j.finished ? 'FINISHED' : 'RUNNING';
  });

  // link
  rows.append('div').classed('cell', true)
      .append('a').attr('href', function(datum, index) {
        return '/tests/' + datum.uuid;
      })
      .text('link');
}
