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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.vfs.LocalPath.MacOsPathPolicy;
import com.google.devtools.build.lib.vfs.LocalPath.OsPathPolicy;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests Mac-specific parts of {@link LocalPath}. */
@RunWith(JUnit4.class)
public class MacOsLocalPathTest extends UnixLocalPathTest {

  @Override
  protected OsPathPolicy getFilePathOs() {
    return new MacOsPathPolicy();
  }

  @Test
  public void testMacEqualsAndHashCode() {
    new EqualsTester()
        .addEqualityGroup(create("a/b"), create("A/B"))
        .addEqualityGroup(create("/a/b"), create("/A/B"))
        .addEqualityGroup(create("something/else"))
        .addEqualityGroup(create("/something/else"))
        .testEquals();
  }

  @Test
  public void testCaseIsPreserved() {
    assertThat(create("a/B").getPathString()).isEqualTo("a/B");
  }
}
