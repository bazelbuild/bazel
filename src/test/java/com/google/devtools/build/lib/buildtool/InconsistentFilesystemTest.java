// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Simple test that Blaze is resilient to an IOException that incorrectly indicates the parent
 * exists (but is not a directory) when trying to create an output file's parent directory.
 */
@RunWith(JUnit4.class)
public class InconsistentFilesystemTest extends BuildIntegrationTestCase {

  @Override
  protected boolean realFileSystem() {
    // Must have real filesystem for MockTools to give us an environment we can execute actions in.
    return true;
  }

  @Override
  protected FileSystem createFileSystem() {
    return new UnixFileSystem(DigestHashFunction.getDefaultUnchecked()) {
      boolean threwException = false;

      @Override
      public boolean createDirectory(Path path) throws IOException {
        String pathString = path.getPathString();
        if (pathString.endsWith("foo") && pathString.contains("blaze-out") && !threwException) {
          threwException = true;
          throw new IOException(path.getPathString() + " (File exists)");
        }
        return super.createDirectory(path);
      }
    };
  }

  @Test
  public void testOutputDirFirstThrowsThenDoesntExist() throws Exception {
    write("foo/BUILD", "genrule(name = 'foo', outs = ['out'], cmd = 'touch $@')");
    buildTarget("//foo:foo");
  }
}
