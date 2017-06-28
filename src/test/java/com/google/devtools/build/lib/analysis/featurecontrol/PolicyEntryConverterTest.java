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

package com.google.devtools.build.lib.analysis.featurecontrol;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the policy entry option converter. */
@RunWith(JUnit4.class)
public final class PolicyEntryConverterTest {

  @Test
  public void convert_successfullyConvertsValidPair() throws Exception {
    assertThat(new PolicyEntryConverter().convert("jeepers=//creepers:peepers"))
        .isEqualTo(PolicyEntry.create("jeepers", Label.parseAbsolute("//creepers:peepers")));
  }

  @Test
  public void convert_successfullyConvertsValidLabelWithEqualsSign() throws Exception {
    assertThat(new PolicyEntryConverter().convert("slamjam=//whoomp:it=there"))
        .isEqualTo(PolicyEntry.create("slamjam", Label.parseAbsolute("//whoomp:it=there")));
  }

  @Test
  public void convert_acceptsBarePackageNameAsDefaultTarget() throws Exception {
    assertThat(new PolicyEntryConverter().convert("two=//go"))
        .isEqualTo(PolicyEntry.create("two", Label.parseAbsolute("//go:go")));
  }

  @Test
  public void convert_acceptsRepositoryPrefixedLabel() throws Exception {
    assertThat(new PolicyEntryConverter().convert("here=@somewhere//else:where"))
        .isEqualTo(PolicyEntry.create("here", Label.parseAbsolute("@somewhere//else:where")));
  }

  @Test
  public void convert_acceptsBareRepository() throws Exception {
    assertThat(new PolicyEntryConverter().convert("aliens=@space"))
        .isEqualTo(PolicyEntry.create("aliens", Label.parseAbsolute("@space//:space")));
  }

  @Test
  public void convert_failsToConvertWithoutDivider() throws Exception {
    try {
      new PolicyEntryConverter().convert("hack//sign:on");
      fail("Expected an exception.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("missing =");
    }
  }

  @Test
  public void convert_failsToConvertLabelAlone() throws Exception {
    try {
      new PolicyEntryConverter().convert("//plain:label");
      fail("Expected an exception.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("missing =");
    }
  }

  @Test
  public void convert_failsToConvertFeatureIdAlone() throws Exception {
    try {
      new PolicyEntryConverter().convert("plainFeature");
      fail("Expected an exception.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("missing =");
    }
  }

  @Test
  public void convert_failsToConvertEmptyFeature() throws Exception {
    try {
      new PolicyEntryConverter().convert("=//R1:C1");
      fail("Expected an exception.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("feature cannot be empty");
    }
  }

  @Test
  public void convert_failsToConvertEmptyLabel() throws Exception {
    try {
      new PolicyEntryConverter().convert("warp=");
      fail("Expected an exception.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("label cannot be empty");
    }
  }

  @Test
  public void convert_failsToConvertInvalidLabel() throws Exception {
    try {
      new PolicyEntryConverter().convert("wrong=//wrong:wrong//wrong");
      fail("Expected an exception.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat()
          .contains("target names may not contain '//' path separators");
    }
  }

  @Test
  public void convert_failsToConvertRelativeLabel() throws Exception {
    try {
      new PolicyEntryConverter().convert("wrong=wrong:wrong");
      fail("Expected an exception.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat()
          .contains("invalid label: wrong:wrong");
    }
  }

  @Test
  public void convert_failsToConvertFilesystemPathLabel() throws Exception {
    try {
      new PolicyEntryConverter().convert("wrong=/usr/local/google/tmp/filename.ext");
      fail("Expected an exception.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat()
          .contains("invalid label: /usr/local/google/tmp/filename.ext");
    }
  }
}
