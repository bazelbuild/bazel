// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.Target;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link com.google.devtools.build.lib.buildtool.InstrumentationFilterSupport}. */
@RunWith(JUnit4.class)
public class InstrumentationFilterSupportTest extends BuildViewTestCase {

  @Test
  public void testComputeInstrumentationFilter() throws Exception {
    EventCollector events = new EventCollector(EventKind.INFO);
    scratch.file("foo/BUILD", "filegroup(name='t', srcs=['t.sh'])");
    scratch.file("foobar/BUILD", "filegroup(name='t', srcs=['t.sh'])");
    List<Target> listOfTargets = new ArrayList<>();
    listOfTargets.add(getTarget("//foo:t"));
    listOfTargets.add(getTarget("//foobar:t"));
    Collection<Target> targets = Collections.unmodifiableCollection(listOfTargets);
    String expectedFilter = "^//foo[/:],^//foobar[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
  }
}
