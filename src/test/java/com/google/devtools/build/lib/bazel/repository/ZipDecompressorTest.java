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

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ZipDecompressor}.
 */
@RunWith(JUnit4.class)
public class ZipDecompressorTest {

  private static final int FILE = 0100644;
  private static final int EXECUTABLE = 0100755;
  private static final int DIRECTORY = 040755;

  // External attributes hold the permissions in the higher-order bits, so the input int has to be
  // shifted.
  private static final int FILE_ATTRIBUTE = FILE << 16;
  private static final int EXECUTABLE_ATTRIBUTE = EXECUTABLE << 16;
  private static final int DIRECTORY_ATTRIBUTE = DIRECTORY << 16;

  @Test
  public void testGetPermissions() throws Exception {
    int permissions = ZipDecompressor.getPermissions(FILE_ATTRIBUTE, "foo/bar");
    assertThat(permissions).isEqualTo(FILE);
    permissions = ZipDecompressor.getPermissions(EXECUTABLE_ATTRIBUTE, "foo/bar");
    assertThat(permissions).isEqualTo(EXECUTABLE);
    permissions = ZipDecompressor.getPermissions(DIRECTORY_ATTRIBUTE, "foo/bar");
    assertThat(permissions).isEqualTo(DIRECTORY);
  }

  @Test
  public void testWindowsPermissions() throws Exception {
    int permissions = ZipDecompressor.getPermissions(ZipDecompressor.WINDOWS_DIRECTORY, "foo/bar");
    assertThat(permissions).isEqualTo(DIRECTORY);
    permissions = ZipDecompressor.getPermissions(ZipDecompressor.WINDOWS_FILE, "foo/bar");
    assertThat(permissions).isEqualTo(EXECUTABLE);
  }

  @Test
  public void testDirectoryWithRegularFilePermissions() throws Exception {
    int permissions = ZipDecompressor.getPermissions(FILE, "foo/bar/");
    assertThat(permissions).isEqualTo(040755);
  }
}
