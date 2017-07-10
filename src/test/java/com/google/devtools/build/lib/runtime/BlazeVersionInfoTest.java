// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static java.util.Collections.singletonMap;

import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.util.StringUtilities;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link BlazeVersionInfo}.
 */
@RunWith(JUnit4.class)
public class BlazeVersionInfoTest {

  @Test
  public void testEmptyVersionInfoMeansNotAvailable() {
    BlazeVersionInfo info = new BlazeVersionInfo(Collections.<String, String>emptyMap());
    assertThat(info.isAvailable()).isFalse();
    assertThat(info.getSummary()).isNull();
    assertThat(info.getReleaseName()).isEqualTo("development version");
  }

  @Test
  public void testReleaseNameIsDevelopmentIfBuildLabelIsNull() {
    Map<String, String> data = singletonMap("Build label", "");
    BlazeVersionInfo info = new BlazeVersionInfo(data);
    assertThat(info.getReleaseName()).isEqualTo("development version");
  }

  @Test
  public void testReleaseNameIfBuildLabelIsPresent() {
    Map<String, String> data = singletonMap("Build label", "3/4/2009 (gold)");
    BlazeVersionInfo info = new BlazeVersionInfo(data);
    assertThat(info.getReleaseName()).isEqualTo("release 3/4/2009 (gold)");
  }

  @Test
  public void testFancySummaryFormatting() {
    Map<String, String> data = new HashMap<>();
    data.put("Some entry", "foo");
    data.put("Another entry", "bar");
    data.put("And a third entry", "baz");
    BlazeVersionInfo info = new BlazeVersionInfo(data);
    Map<String, String> sortedData = new TreeMap<>(data);
    assertThat(info.getSummary()).isEqualTo(StringUtilities.layoutTable(sortedData));
  }
}
