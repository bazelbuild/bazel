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


import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link BlazeDirectories}. */
@RunWith(JUnit4.class)
public class BlazeDirectoriesTest extends FoundationTestCase {

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            new BlazeDirectories(
                new ServerDirectories(
                    FsUtils.TEST_FILESYSTEM.getPath("/install_base"),
                    FsUtils.TEST_FILESYSTEM.getPath("/output_base"),
                    FsUtils.TEST_FILESYSTEM.getPath("/user_root")),
                FsUtils.TEST_FILESYSTEM.getPath("/workspace"),
                /* defaultSystemJavabase= */ null,
                TestConstants.PRODUCT_NAME),
            new BlazeDirectories(
                new ServerDirectories(
                    FsUtils.TEST_FILESYSTEM.getPath("/install_base"),
                    FsUtils.TEST_FILESYSTEM.getPath("/output_base"),
                    FsUtils.TEST_FILESYSTEM.getPath("/user_root"),
                    FsUtils.TEST_FILESYSTEM.getPath("/output_base/execroot"),
                    "1234abcd1234abcd1234abcd1234abcd"),
                FsUtils.TEST_FILESYSTEM.getPath("/workspace"),
                /* defaultSystemJavabase= */ null,
                TestConstants.PRODUCT_NAME),
            new BlazeDirectories(
                new ServerDirectories(
                    FsUtils.TEST_FILESYSTEM.getPath("/install_base"),
                    FsUtils.TEST_FILESYSTEM.getPath("/output_base"),
                    FsUtils.TEST_FILESYSTEM.getPath("/user_root")),
                FsUtils.TEST_FILESYSTEM.getPath("/workspace"),
                /* defaultSystemJavabase= */ null,
                TestConstants.PRODUCT_NAME))
        .addDependency(FileSystem.class, FsUtils.TEST_FILESYSTEM)
        .runTests();
  }

  @Test
  public void testBlazeExecIsNullInBazel() {
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(
                FsUtils.TEST_FILESYSTEM.getPath("/install_base"),
                FsUtils.TEST_FILESYSTEM.getPath("/output_base"),
                FsUtils.TEST_FILESYSTEM.getPath("/user_root")),
            FsUtils.TEST_FILESYSTEM.getPath("/workspace"),
            /* defaultSystemJavabase= */ null,
            /* productName = */ "bazel");
    assertThat(directories.getBlazeExecRoot()).isNull();
    assertThat(directories.getBlazeOutputPath()).isNull();
  }
}
