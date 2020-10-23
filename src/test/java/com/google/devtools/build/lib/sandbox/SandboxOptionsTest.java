// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code SandboxOptions}. */
@RunWith(JUnit4.class)
public final class SandboxOptionsTest {

  private ImmutableMap.Entry<String, String> pathPair;

  @Test
  public void testParsingAdditionalMounts_singlePathWithoutColonSucess() throws Exception {
    String source = "/a/bc/def/gh";
    String target = source;
    String input = source;
    pathPair = new SandboxOptions.MountPairConverter().convert(input);
    assertMountPair(pathPair, source, target);
  }

  @Test
  public void testParsingAdditionalMounts_singlePathWithColonSucess() throws Exception {
    String source = "/a/b:c/def/gh";
    String target = source;
    String input = "/a/b\\:c/def/gh";
    pathPair = new SandboxOptions.MountPairConverter().convert(input);
    assertMountPair(pathPair, source, target);
  }

  @Test
  public void testParsingAdditionalMounts_pathPairWithoutColonSucess() throws Exception {
    String source = "/a/bc/def/gh";
    String target = "/1/2/3/4/5";
    String input = source + ":" + target;
    pathPair = new SandboxOptions.MountPairConverter().convert(input);
    assertMountPair(pathPair, source, target);
  }

  @Test
  public void testParsingAdditionalMounts_pathPairWithColonSucess() throws Exception {
    String source = "/a:/bc:/d:ef/gh";
    String target = ":/1/2/3/4/5";
    String input = "/a\\:/bc\\:/d\\:ef/gh:\\:/1/2/3/4/5";
    pathPair = new SandboxOptions.MountPairConverter().convert(input);
    assertMountPair(pathPair, source, target);
  }

  @Test
  public void testParsingAdditionalMounts_tooManyPaths() throws Exception {
    String input = "a/bc/def/gh:/1/2/3:x/y/z";
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> pathPair = new SandboxOptions.MountPairConverter().convert(input));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Input must be a single path to mount inside the sandbox or "
                + "a mounting pair in the form of 'source:target'");
  }

  @Test
  public void testParsingAdditionalMounts_emptyInput() throws Exception {
    String input = "";
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> pathPair = new SandboxOptions.MountPairConverter().convert(input));
    assertThat(
            "Input "
                + input
                + " contains one or more empty paths. "
                + "Input must be a single path to mount inside the sandbox or "
                + "a mounting pair in the form of 'source:target'")
        .isEqualTo(e.getMessage());
  }

  private static void assertMountPair(
      ImmutableMap.Entry<String, String> pathPair, String source, String target) {
    assertThat(source).isEqualTo(pathPair.getKey());
    assertThat(target).isEqualTo(pathPair.getValue());
  }
}
