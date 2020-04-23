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

import com.google.devtools.build.lib.buildtool.util.GoogleBuildIntegrationTestCase;
import com.google.devtools.build.lib.skyframe.MutableSupplier;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/** Abstract test class for tests that want to be aware of filesystem operations. */
@TestSpec(size = Suite.MEDIUM_TESTS)
public abstract class IoHookTestCase extends GoogleBuildIntegrationTestCase {

  /** Type of path operation. */
  protected static enum PathOp {
    MD5_DIGEST,
    FAST_DIGEST,
    STAT,
    CHMOD
  }

  protected boolean fastDigest;

  @Override
  protected boolean realFileSystem() {
    // Must have real filesystem for MockTools to give us an environment we can execute actions in.
    return true;
  }

  /** Listens for file operations. */
  protected interface FileListener {
    void handle(PathOp op, Path path) throws IOException;
  }

  private static final FileListener DUMMY_LISTENER = new FileListener() {
    @Override
    public void handle(PathOp op, Path path) {}
  };

  private MutableSupplier<FileListener> listener = new MutableSupplier<>();

  protected void setListener(FileListener listener) {
    this.listener.set(listener);
  }

  @Override
  protected FileSystem createFileSystem() {
    setListener(DUMMY_LISTENER);
    return new UnixFileSystem(DigestHashFunction.getDefaultUnchecked()) {
      @Override
      protected void chmod(Path path, int chmod) throws IOException {
        listener.get().handle(PathOp.CHMOD, path);
        super.chmod(path, chmod);
      }

      @Override
      protected FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
        listener.get().handle(PathOp.STAT, path);
        return super.statIfFound(path, followSymlinks);
      }

      @Override
      protected UnixFileStatus statInternal(Path path, boolean followSymlinks) throws IOException {
        listener.get().handle(PathOp.STAT, path);
        return super.statInternal(path, followSymlinks);
      }

      @Override
      protected byte[] getDigest(Path path) throws IOException {
        listener.get().handle(PathOp.MD5_DIGEST, path);
        return super.getDigest(path);
      }

      @Override
      protected byte[] getFastDigest(Path path) throws IOException {
        listener.get().handle(PathOp.FAST_DIGEST, path);
        // Importantly, listener.get().handle(PathOp.MD5_DIGEST, path) is not called here.
        return fastDigest ? super.getDigest(path) : null;
      }
    };
  }
}
