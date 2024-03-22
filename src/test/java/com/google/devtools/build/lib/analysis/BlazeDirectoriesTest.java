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
package com.google.devtools.build.lib.analysis;

import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link BlazeDirectories}. */
@RunWith(JUnit4.class)
public final class BlazeDirectoriesTest extends FoundationTestCase {

  private final BlazeDirectories directories =
      new BlazeDirectories(
          new ServerDirectories(
              FsUtils.TEST_FILESYSTEM.getPath("/install_base"),
              FsUtils.TEST_FILESYSTEM.getPath("/output_base"),
              FsUtils.TEST_FILESYSTEM.getPath("/user_root")),
          FsUtils.TEST_FILESYSTEM.getPath("/workspace"),
          /* defaultSystemJavabase= */ null,
          /* productName= */ "bazel");

  @Test
  public void noBlazeExecRootInBazel() {
    assertThrows(NullPointerException.class, directories::getBlazeExecRoot);
  }

  @Test
  public void noBlazeOutputPathInBazel() {
    assertThrows(NullPointerException.class, directories::getBlazeOutputPath);
  }
}
