// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.server.FailureDetails.PackageOptions.Code;
import com.google.devtools.build.lib.util.AbruptExitException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@code --incompatible_source_directory_is_directory} without source directory tracking. */
@RunWith(JUnit4.class)
public final class SourceDirectoryTrackingDisabledTest extends BuildIntegrationTestCase {

  @Test
  public void sourceDirectoryIsDirectory_failsWithoutTracking() throws Exception {
    write("foo/BUILD", "filegroup(name = 'foo')");
    addOptions("--incompatible_source_directory_is_directory");

    AbruptExitException e =
        assertThrows(AbruptExitException.class, () -> buildTarget("//foo"));

    assertThat(e.getDetailedExitCode().getFailureDetail().getPackageOptions().getCode())
        .isEqualTo(Code.SOURCE_DIRECTORY_TRACKING_DISABLED);
  }

  @Test
  public void withoutFlag_succeeds() throws Exception {
    write("foo/BUILD", "filegroup(name = 'foo')");

    buildTarget("//foo");
  }
}
