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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.io.MoreFiles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidDataBindingProcessingAction}. */
@RunWith(JUnit4.class)
public class AndroidDataBindingProcessingActionTest {

  private final String testDataPrefix =
      System.getProperty("AndroidDataBindingProcessingActionTest.testDataPrefix", "");

  private Path tempDir;
  private Path dataBindingInfoOut;

  @Before
  public void setupPaths() throws Exception {
    tempDir = Files.createTempDirectory(toString());
    dataBindingInfoOut = tempDir.resolve("layout-info.zip");
  }

  @After
  public void deletePaths() throws Exception {
    Files.deleteIfExists(dataBindingInfoOut);
    MoreFiles.deleteRecursively(tempDir);
  }

  @Test
  public void testNoResourceRoots() throws Exception {

    String[] args = {
        "--dataBindingInfoOut=" + dataBindingInfoOut,
        "--appId=foo.bar",
    };
    AndroidDataBindingProcessingAction.main(args);

    assertThat(Files.exists(dataBindingInfoOut)).isTrue();

    ZipFile layoutInfo = new ZipFile(dataBindingInfoOut.toFile());
    assertThat(layoutInfo.entries().hasMoreElements()).isFalse();
  }

  @Test
  public void testOneResourceRoot() throws Exception {

    String resourceRoot =
        testDataPrefix + "src/test/java/com/google/devtools/build/android/testing/databinding/res";

    String[] args = {
        "--resource_root=" + resourceRoot,
        "--output_resource_root=" + tempDir.resolve("res"),
        "--dataBindingInfoOut=" + dataBindingInfoOut,
        "--appId=foo.bar",
    };
    AndroidDataBindingProcessingAction.main(args);

    assertThat(Files.exists(dataBindingInfoOut)).isTrue();

    ZipFile layoutInfo = new ZipFile(dataBindingInfoOut.toFile());
    List<? extends ZipEntry> zipEntries = Collections.list(layoutInfo.entries());
    assertThat(zipEntries).hasSize(1);
  }

  @Test
  public void testTwoResourceRoots() throws Exception {

    String resourceRoot =
        testDataPrefix + "src/test/java/com/google/devtools/build/android/testing/databinding/res";
    String resourceRoot2 =
        testDataPrefix + "src/test/java/com/google/devtools/build/android/testing/databinding/res2";

    String[] args = {
        "--resource_root=" + resourceRoot,
        "--output_resource_root=" + tempDir.resolve("res"),
        "--resource_root=" + resourceRoot2,
        "--output_resource_root=" + tempDir.resolve("res2"),
        "--dataBindingInfoOut=" + dataBindingInfoOut,
        "--appId=foo.bar",
    };
    AndroidDataBindingProcessingAction.main(args);

    assertThat(Files.exists(dataBindingInfoOut)).isTrue();

    ZipFile layoutInfo = new ZipFile(dataBindingInfoOut.toFile());
    List<? extends ZipEntry> zipEntries = Collections.list(layoutInfo.entries());
    assertThat(zipEntries).hasSize(2);
  }

  @Test
  public void testInputOutputResourceRootsMismatchThrows() throws Exception {

    // resource_root, no output_resource_root
    String[] args1 = {
        "--resource_root=foo",
        "--dataBindingInfoOut=" + dataBindingInfoOut,
        "--appId=foo.bar",
    };

    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> AndroidDataBindingProcessingAction.main(args1));

    // output_resource_root, no resource_root
    String[] args2 = {
        "--output_resource_root=foo",
        "--dataBindingInfoOut=" + dataBindingInfoOut,
        "--appId=foo.bar",
    };

    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> AndroidDataBindingProcessingAction.main(args2));

    // 2 resource_roots, but 1 output_resource_root
    String[] args3 = {
        "--resource_root=foo",
        "--output_resource_root=bar",
        "--resource_root=baz",
        "--dataBindingInfoOut=" + dataBindingInfoOut,
        "--appId=foo.bar",
    };

    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> AndroidDataBindingProcessingAction.main(args3));

    // 2 output_resource_root, but 1 resource_root
    String[] args4 = {
        "--resource_root=foo",
        "--output_resource_root=bar",
        "--output_resource_root=baz",
        "--dataBindingInfoOut=" + dataBindingInfoOut,
        "--appId=foo.bar",
    };

    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> AndroidDataBindingProcessingAction.main(args4));
  }
}
