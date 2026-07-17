// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import java.util.Map;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link VersionCommand#getInfo}. */
@RunWith(JUnit4.class)
public class VersionCommandTest {
  private static final boolean GNU_FORMAT = true;
  private static final boolean LEGACY_FORMAT = false;

  @Test
  public void testNoSummary() {
    assertThat(
            VersionCommand.getInfo(
                "product", new BlazeVersionInfo(ImmutableMap.of()), LEGACY_FORMAT))
        .isEmpty();
  }

  @Test
  public void testNoSummaryGnuFormat() {
    assertThat(
            VersionCommand.getInfo("product", new BlazeVersionInfo(ImmutableMap.of()), GNU_FORMAT))
        .isEmpty();
  }

  @Test
  public void testNoVersionGnuFormat() {
    Map<String, String> map = ImmutableMap.of(BlazeVersionInfo.BUILD_LABEL, "");
    Optional<String> info =
        VersionCommand.getInfo("product", new BlazeVersionInfo(map), GNU_FORMAT);
    assertThat(info).hasValue("product no_version");
  }

  @Test
  public void testVersionGnuFormat() {
    Map<String, String> map = ImmutableMap.of(BlazeVersionInfo.BUILD_LABEL, "1.2");
    Optional<String> info =
        VersionCommand.getInfo("product", new BlazeVersionInfo(map), GNU_FORMAT);
    assertThat(info).hasValue("product 1.2");
  }

  @Test
  public void testLegacyFormat() {
    Map<String, String> map =
        ImmutableMap.of(BlazeVersionInfo.BUILD_LABEL, "version", "More", "foo");
    Optional<String> info =
        VersionCommand.getInfo("product", new BlazeVersionInfo(map), LEGACY_FORMAT);
    assertThat(info).hasValue("Build label: version\nMore: foo");
  }
}
