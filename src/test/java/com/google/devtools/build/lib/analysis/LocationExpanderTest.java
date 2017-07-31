// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link LocationExpander}. */
@RunWith(JUnit4.class)
public class LocationExpanderTest extends BuildViewTestCase {

  @Before
  public void createFiles() throws Exception {
    // Set up a rule to test expansion in.
    scratch.file("files/fileA");
    scratch.file("files/fileB");

    scratch.file(
        "files/BUILD",
        "filegroup(name='files',",
        "  srcs = ['fileA', 'fileB'])",
        "sh_library(name='lib',",
        "  deps = [':files'])");
  }

  private LocationExpander makeExpander(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    RuleContext ruleContext = getRuleContext(target);
    return new LocationExpander(ruleContext);
  }

  @Test
  public void location_absolute() throws Exception {
    LocationExpander expander = makeExpander("//files:files");
    String input = "foo $(location //files:fileA) bar";
    String result = expander.expand(input);

    assertThat(result).isEqualTo("foo files/fileA bar");
  }

  @Test
  public void locations_spaces() throws Exception {
    scratch.file("spaces/file with space A");
    scratch.file("spaces/file with space B");
    scratch.file(
        "spaces/BUILD",
        "filegroup(name='files',",
        "  srcs = ['file with space A', 'file with space B'])",
        "sh_library(name='lib',",
        "  deps = [':files'])");

    LocationExpander expander = makeExpander("//spaces:lib");
    String input = "foo $(locations :files) bar";
    String result = expander.expand(input);

    assertThat(result).isEqualTo("foo 'spaces/file with space A' 'spaces/file with space B' bar");
  }

  @Test
  public void location_relative() throws Exception {
    LocationExpander expander = makeExpander("//files:files");
    String input = "foo $(location :fileA) bar";
    String result = expander.expand(input);

    assertThat(result).isEqualTo("foo files/fileA bar");
  }

  @Test
  public void locations_relative() throws Exception {
    LocationExpander expander = makeExpander("//files:lib");
    String input = "foo $(locations :files) bar";
    String result = expander.expand(input);

    assertThat(result).isEqualTo("foo files/fileA files/fileB bar");
  }
}
