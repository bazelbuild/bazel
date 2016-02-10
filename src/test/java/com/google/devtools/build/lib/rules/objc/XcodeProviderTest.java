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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.objc.XcodeProvider.xcodeTargetName;

import com.google.devtools.build.lib.cmdline.Label;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for XcodeProvider.
 */
@RunWith(JUnit4.class)
public class XcodeProviderTest {
  @Test
  public void testXcodeTargetName() throws Exception {
    assertThat(xcodeTargetName(Label.parseAbsolute("//foo:bar"))).isEqualTo("bar_foo");
    assertThat(xcodeTargetName(Label.parseAbsolute("//foo/bar:baz"))).isEqualTo("baz_bar_foo");
  }

  @Test
  public void testExternalXcodeTargetName() throws Exception {
    assertThat(xcodeTargetName(Label.parseAbsolute("@repo_name//foo:bar")))
        .isEqualTo("bar_external_repo_name_foo");
  }
}
