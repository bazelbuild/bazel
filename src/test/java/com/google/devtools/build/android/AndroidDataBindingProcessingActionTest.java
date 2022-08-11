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
import com.google.common.io.RecursiveDeleteOption;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/** Tests for {@link AndroidDataBindingProcessingAction}. */
@RunWith(JUnit4.class)
public class AndroidDataBindingProcessingActionTest {

  private final Path testDataPrefix =
      Paths.get(System.getenv("TEST_BINARY")).resolveSibling("testing");

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
    MoreFiles.deleteRecursively(tempDir, RecursiveDeleteOption.ALLOW_INSECURE);
  }

  @Test
  public void testNoResourceRoots() throws Exception {

    String[] args = {
      "--output_resource_directory=" + tempDir.resolve("res"),
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

    String resourceRoot = testDataPrefix.resolve("databinding/res").toString();

    String[] args = {
      "--output_resource_directory=" + tempDir.resolve("res"),
      "--resource_root=" + resourceRoot,
      "--dataBindingInfoOut=" + dataBindingInfoOut,
      "--appId=foo.bar",
    };
    AndroidDataBindingProcessingAction.main(args);

    assertThat(Files.exists(dataBindingInfoOut)).isTrue();

    ZipFile layoutInfo = new ZipFile(dataBindingInfoOut.toFile());
    List<? extends ZipEntry> zipEntries = Collections.list(layoutInfo.entries());
    assertThat(zipEntries).hasSize(1);
    assertThat(
            zipEntries.stream()
                .allMatch(
                    entry ->
                        entry.getLastModifiedTime().toMillis()
                            == AarGeneratorAction.DEFAULT_TIMESTAMP))
        .isTrue();
  }

  @Test
  public void testTwoResourceRoots() throws Exception {
    String resourceRoot = testDataPrefix.resolve("databinding/res").toString();
    String resourceRoot2 = testDataPrefix.resolve("databinding/res2").toString();

    String[] args = {
      "--output_resource_directory=" + tempDir.resolve("res"),
      "--resource_root=" + resourceRoot,
      "--resource_root=" + resourceRoot2,
      "--dataBindingInfoOut=" + dataBindingInfoOut,
      "--appId=foo.bar",
    };
    AndroidDataBindingProcessingAction.main(args);

    assertThat(Files.exists(dataBindingInfoOut)).isTrue();

    ZipFile layoutInfo = new ZipFile(dataBindingInfoOut.toFile());
    List<? extends ZipEntry> zipEntries = Collections.list(layoutInfo.entries());
    assertThat(zipEntries).hasSize(2);
  }
}
