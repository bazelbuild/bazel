// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;

import com.google.devtools.build.lib.vfs.util.FileSystems;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * This class handles the tests for the FileSystems class.
 */
@RunWith(JUnit4.class)
public class FileSystemsTest {

  @Test
  public void testFileSystemsCreatesOnlyOneDefaultNative() {
    assertSame(FileSystems.getNativeFileSystem(),
               FileSystems.getNativeFileSystem());
  }

  @Test
  public void testFileSystemsCreatesOnlyOneDefaultJavaIo() {
    assertSame(FileSystems.getJavaIoFileSystem(),
               FileSystems.getJavaIoFileSystem());
  }

  @Test
  public void testFileSystemsCanSwitchDefaults() {
    assertNotSame(FileSystems.getNativeFileSystem(),
                  FileSystems.getJavaIoFileSystem());
  }
}
