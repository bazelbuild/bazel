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

package com.google.devtools.build.lib.sandbox;

import static junit.framework.TestCase.fail;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RealSandboxfsProcess}. */
@RunWith(JUnit4.class)
public class RealSandboxfsProcessTest extends BaseSandboxfsProcessTest {

  @Override
  Path newTmpDir() {
    String rawTmpDir = System.getenv("TEST_TMPDIR");
    if (rawTmpDir == null) {
      fail("Test requires TEST_TMPDIR to be defined in the environment");
    }

    FileSystem fileSystem = new JavaIoFileSystem(DigestHashFunction.DEFAULT_HASH_FOR_TESTS);
    Path tmpDir = fileSystem.getPath(rawTmpDir);
    if (!tmpDir.isDirectory()) {
      fail("TEST_TMPDIR must point to a directory");
    }
    return tmpDir;
  }

  @Override
  SandboxfsProcess mount(Path mountPoint) throws IOException {
    String rawSandboxfs = System.getenv("SANDBOXFS");
    if (rawSandboxfs == null) {
      fail("Test requires SANDBOXFS to be defined in the environment");
    }

    FileSystem fileSystem = new JavaIoFileSystem(DigestHashFunction.DEFAULT_HASH_FOR_TESTS);
    Path sandboxfs = fileSystem.getPath(rawSandboxfs);
    if (!sandboxfs.isExecutable()) {
      fail("SANDBOXFS must point to an executable binary");
    }
    return RealSandboxfsProcess.mount(
        sandboxfs.asFragment(), mountPoint, fileSystem.getPath("/dev/stderr"));
  }
}
