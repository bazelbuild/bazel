var icons = {
  running: '?',
  passed: '\u2705',
  errors: '\u274c'
};


function showData() {
  renderDetails(getDetailsData(), false);
  renderInfo(getCommandInfo());
}

function getCommandInfo() {
  var url = document.URL;
  if (url[url.length - 1] != '/') {
    url += '/';
  }
  return $.ajax({
      type: 'GET',
      url: url + 'info',
      async: false
  }).responseJSON;
}

function getDetailsData() {
  // TODO(bazel-team): auto refresh, async callback
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


function showDate(d) {
  function pad(x) {
    return x < 10 ? '0' + x : '' + x;
  }
  var today = new Date();
  var result = pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' +
               pad(d.getSeconds());
  if (d.getDate() === today.getDate() && d.getMonth() === today.getMonth() &&
      d.getYear() === today.getYear()) {
    result += ' today';
  } else {
    result += pad(d.getDate()) + ' ' + pad(d.getMonth()) + ' ' + d.getYear();
  }
  return result;
}

function renderInfo(info) {
  $('#testInfo').empty();
  var data = [
    ['Targets: ', info['targets']],
    ['Started at: ', showDate(new Date(info['startTime']))],
  ];
  if (info['finished']) {
    data.push(['Finished at: ', showDate(new Date(info['endTime']))]);
  } else {
    data.push(['Still running']);
  }
  var selection = d3.select('#testInfo').selectAll()
      .data(data)
      .enter().append('div')
      .classed('info-cell', true);
  selection
      .append('div')
      .classed('info-detail', true)
      .text(function(d) { return d[0]; });
  selection
      .append('div')
      .classed('info-detail', true)
      .text(function(d) { return d[1]; });
}

// predicate is either a predicate function or null - in the latter case
// everything is shown
function renderDetails(tests, predicate) {
  $('#testDetails').empty();
  if (tests.length == 0) {
    $('#testDetails').text('No test details to display.');
    return;
  }
  // flatten object to array and set visibility
  tests = $.map(tests, function(element) {
    if (predicate) {
      setVisibility(predicate, element);
    }
    return element;
  });
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
      var result = '';
      var failures = [];
      var errors = [];
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
      if (result == '') {
        return j.status;
      }
      return result;
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
                  var childStatus = propagateStatus(target);
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
    function testTargetStatusIcon() {
      selection.append('div')
          .classed('test-detail', true)
          .attr('color', function(target) {
                  var childStatus = propagateStatus(target);
                  if (target.finished == false) {
                    return 'running';
                  } else {
                    if (childStatus == 'PASSED') {
                      return 'passed';
                    } else {
                      return 'errors';
                    }
                  }})
          .text(function(target) {
                  var childStatus = propagateStatus(target);
                  if (target.finished == false) {
                    return icons.running;
                  } else {
                    if (childStatus == 'PASSED') {
                      return icons.passed;
                    } else {
                      return icons.errors;
                    }
                  }
          });
    }
    function testCaseTime() {
      selection.append('div').classed('test-detail', true).text(function(j) {
        var times = $.map(j.times, function(element, key) { return element });
        if (times.length < 1) {
          return '?';
        } else {
          return Math.max.apply(Math, times) / 1000 + ' s';
        }
      });
    }

    function visibilityFilter() {
      selection.attr('show', function(datum) {
        return ('show' in datum) ? datum['show'] : true;
      });
    }

    // Toplevel nodes represent test targets, so they look a bit different
    if (toplevel) {
      testTargetStatusIcon();
      fullName();
    } else {
      testTargetStatusIcon();
      fullName();
      testCaseStatus();
      testCaseTime();
    }
    visibilityFilter();
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
        .on('click', function(datum) {
          if ($(this).attr('toggle') == 'on') {
            $(this).siblings('.test-case').not('[show=false]').hide();
            $(this).attr('toggle', 'off');
            $(this).text('Show details');
          } else {
            $(this).siblings('.test-case').not('[show=false]').show();
            $(this).attr('toggle', 'on');
            $(this).text('Hide details');
          }
        });
    nextLevelNodes.enter().append('div').classed('test-case', true);
    addNestedDetails(nextLevelNodes, false);
  }

  addNestedDetails(rows, true);
  $('.button').siblings('.test-case').hide();
  if (predicate) {
    toggleVisibility();
  }
}

function toggleVisibility() {
  $('#testDetails > [show=false]').hide();
  $('#testDetails > [show=true]').show();
  $('[toggle=on]').siblings('[show=false]').hide();
  $('[toggle=on]').siblings('[show=true]').show();
}

function setVisibility(predicate, object) {
  var show = predicate(object);
  var childrenPredicate = predicate;
  // It rarely makes sense to show a non-leaf node and hide its children, so
  // we just show all children
  if (show) {
    childrenPredicate = function() { return true; };
  }
  if ('children' in object) {
    for (var child in object.children) {
      setVisibility(childrenPredicate, object.children[child]);
      show = object.children[child]['show'] || show;
    }
  }
  object['show'] = show;
}

// given a list of predicates, return a function
function intersectFilters(filterList) {
  var filters = filterList.filter(function(x) { return x });
  return function(x) {
    for (var i = 0; i < filters.length; i++) {
      if (!filters[i](x)) {
        return false;
      }
    }
    return true;
  }
}

function textFilterActive() {
  return $('#search').val();
}

function getTestFilters() {
  var statusFilter = null;
  var textFilter = null;
  var filters = [];
  var passed = $('#boxPassed').prop('checked');
  var failed = $('#boxFailed').prop('checked');
  // add checkbox filters only when necessary (ie. something is unchecked - when
  // everything is checked this means user wants to see everything).
  if (!(passed && failed)) {
    var checkBoxFilters = [];
    if (passed) {
      checkBoxFilters.push(function(object) {
        return object.status == 'PASSED';
      });
    }
    if (failed) {
      checkBoxFilters.push(function(object) {
        return 'status' in object && object.status != 'PASSED';
      });
    }
    filters.push(function(object) {
      return checkBoxFilters.some(function(f) { return f(object); });
    });
  }
  if (textFilterActive()) {
    filters.push(function(object) {
      // TODO(bazel-team): would case insentive search make more sense?
      return ('fullName' in object &&
          object.fullName.indexOf($('#search').val()) != -1);
    });
  }
  return filters;
}

function redraw() {
  renderDetails(getDetailsData(), intersectFilters(getTestFilters()));
}

function updateVisibleCases() {
  var predicate = intersectFilters(getTestFilters());
  var parentCases = d3.selectAll('#testDetails > div').data();
  parentCases.forEach(function(element, index) {
    setVisibility(predicate, element);
  });
  d3.selectAll('.test-detail').attr('show', function(datum) {
    return ('show' in datum) ? datum['show'] : true;
  });
  d3.selectAll('.test-case').attr('show', function(datum) {
    return ('show' in datum) ? datum['show'] : true;
  });
  toggleVisibility();
  if (textFilterActive()) {
    // expand nodes to save some clicking - if user searched for something that
    // is leaf of the tree, she definitely wants to see it
    $('#testDetails > [show=true]').find('[toggle=off]').click();
  }
}

function enableControls() {
  var redrawTimeout = null;
  $('#boxPassed').click(updateVisibleCases);
  $('#boxFailed').click(updateVisibleCases);
  $('#search').keyup(function() {
    clearTimeout(redrawTimeout);
    redrawTimeout = setTimeout(updateVisibleCases, 500);
  });
  $('#clearFilters').click(function() {
    $('#boxPassed').prop('checked', true);
    $('#boxFailed').prop('checked', true);
    $('#search').val('');
    updateVisibleCases();
  });
}

$(function() {
  showData();
  enableControls();
});
