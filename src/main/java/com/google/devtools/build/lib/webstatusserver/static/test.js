function showData() {
  renderDetails(getDetailsData());
}

function getDetailsData() {
  // TODO(bazel-team): apply similar changes to the ones in getSummaryData()
  var url = document.URL;
  if (url[url.length - 1] != '/') {
    url += '/';
  }
  return $.ajax({
      type: 'GET',
      url: url + 'details',
      async: false
  }).responseJSON;
}

function renderDetails(tests) {
  // flatten object to array
  tests = $.map(tests, function(element, key) { return element });
  var rows = d3.select('#testDetails').selectAll()
    .data(tests)
    .enter().append('div')
    .classed('test-case', true);

  function addTestDetail(selection, toplevel) {
    function fullName() {
      selection.append('div').classed('test-detail', true).text(function(j) {
        return j.fullName;
      });
    }
    function propagateStatus(j) {
      var
          result = '',
          failures = [],
          errors = [];
      $.each(j.results, function(key, value) {
        if (value == 'FAILED') {
          failures.push(key);
        }
        if (value == 'ERROR') {
          errors.push(key);
        }
      });
      if (failures.length > 0) {
        var s = failures.length > 1 ? 's' : '';
        result += 'Failed on ' + failures.length + ' shard' + s + ': ' +
                    failures.join();
      }
      if (errors.length > 0) {
        var s = failures.length > 1 ? 's' : '';
        result += 'Errors on ' + errors.length + ' shard' + s + ': ' +
                    errors.join();
      }
      return result != '' ? result : 'PASSED';
    }
    function testCaseStatus() {
      selection.append('div')
          .classed('test-detail', true)
          .text(propagateStatus);
    }
    function testTargetStatus() {
      selection.append('div')
          .classed('test-detail', true)
          .text(function(target) {
                  var
                    childStatus = propagateStatus(target);
                  if (target.finished = false) {
                    return target.status + ' ' + stillRunning;
                  } else {
                    if (childStatus == 'PASSED') {
                      return target.status;
                    } else {
                      return target.status + ' ' + childStatus;
                    }
                  }
          });
    }
    function testCaseTime() {
      selection.append('div').classed('test-detail', true).text(function(j) {
        var
          times = $.map(j.times, function(element, key) { return element });
        if (times.length < 1) {
          return '?';
        } else {
          return Math.max.apply(Math, times);
        }
      });
    }

    // Toplevel nodes represent test targets, so they look a bit different
    if (toplevel) {
      fullName();
      testTargetStatus();
    } else {
      fullName();
      testCaseStatus();
      testCaseTime();
    }
  }

  function addNestedDetails(table, toplevel) {
    table.sort(function(data1, data2) {
      if (data1.fullName < data2.fullName) {
        return -1;
      }
      if (data1.fullName > data2.fullName) {
        return 1;
      }
      return 0;
    });

    addTestDetail(table, toplevel);

    // Add children nodes + show/hide button
    var nonLeafNodes = table.filter(function(data, index) {
      return !($.isEmptyObject(data.children));
    });
    var nextLevelNodes = nonLeafNodes.selectAll().data(function(d) {
      return $.map(d.children, function(element, key) { return element });
    });

    if (nextLevelNodes.enter().empty()) {
      return;
    }

    nonLeafNodes
        .append('div')
        .classed('test-detail', true)
        .classed('button', true)
        .text(function(j) {
          return 'Show details';
        })
        .attr('toggle', 'off')
        .on('click', function() {
          if ($(this).attr('toggle') == 'on') {
            $(this).siblings('.test-case').hide();
            $(this).attr('toggle', 'off');
            $(this).text('Show details');
          } else {
            $(this).siblings().show();
            $(this).attr('toggle', 'on');
            $(this).text('Hide details');
          }
        });
    nextLevelNodes.enter().append('div').classed('test-case', true);
    addNestedDetails(nextLevelNodes, false);
  }

  addNestedDetails(rows, true);
  $('.button').siblings('.test-case').hide();
}
