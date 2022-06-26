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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import java.util.Collections;
import java.util.Map;
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
  public void testGetSummaryReturnsOrderedTablifiedData() {
    ImmutableMap<String, String> data =
        ImmutableMap.of("key3", "foo", "key2", "bar", "key1", "baz");

    BlazeVersionInfo info = new BlazeVersionInfo(data);
    assertThat(info.getSummary()).isEqualTo("key1: baz\nkey2: bar\nkey3: foo");
  }

  @Test
  public void testVersionIsHeadIfBuildLabelIsNull() {
    BlazeVersionInfo info = new BlazeVersionInfo(ImmutableMap.of());
    assertThat(info.getVersion()).isEmpty();
  }

  @Test
  public void testVersionsIIfBuildLabelIsPresent() {
    Map<String, String> data = ImmutableMap.of("Build label", "123.4");
    BlazeVersionInfo info = new BlazeVersionInfo(data);
    assertThat(info.getVersion()).isEqualTo("123.4");
  }
}
