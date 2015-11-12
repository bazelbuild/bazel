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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.RootedPath;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Sanity checks for {@link PrepareDepsOfTargetsUnderDirectoryValue}. */
@RunWith(JUnit4.class)
public class PrepareDepsOfTargetsUnderDirectoryValueTest {

  @Test
  public void testOfMatchingEmptyValueReturnsEmptySingleton() {
    PrepareDepsOfTargetsUnderDirectoryValue val =
        PrepareDepsOfTargetsUnderDirectoryValue.of(false, ImmutableMap.<RootedPath, Boolean>of());
    assertThat(val).isSameAs(PrepareDepsOfTargetsUnderDirectoryValue.EMPTY);
  }

  @Test
  public void testOfMatchingEmptyDirectoryPackageReturnsSingleton() {
    PrepareDepsOfTargetsUnderDirectoryValue val =
        PrepareDepsOfTargetsUnderDirectoryValue.of(true, ImmutableMap.<RootedPath, Boolean>of());
    assertThat(val).isSameAs(PrepareDepsOfTargetsUnderDirectoryValue.EMPTY_DIRECTORY_PACKAGE);
  }
}
