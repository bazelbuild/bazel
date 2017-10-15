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
package com.google.devtools.build.lib.testutil;

import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.File;
import java.io.IOException;

/**
 * An implementation of the FileOutErr that uses an in-memory file behind the scenes.
 */
public class TestFileOutErr extends FileOutErr {

  public TestFileOutErr() {
    super(new FlushingFileRecordingOutputStream(newInMemoryFile("out.log")),
        new FlushingFileRecordingOutputStream(newInMemoryFile("err.log")));

  }

  public TestFileOutErr(Path root) {
    super(
        new FlushingFileRecordingOutputStream(root.getChild("out.log")),
        new FlushingFileRecordingOutputStream(root.getChild("err.log")));
  }

  private static Path newInMemoryFile(File root, String name) {
    InMemoryFileSystem inMemFS = new InMemoryFileSystem();
    Path directory = inMemFS.getPath(root.getPath());
    try {
      FileSystemUtils.createDirectoryAndParents(directory);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    return directory.getRelative(name);
  }

  private static Path newInMemoryFile(String name) {
    return newInMemoryFile(new File("/inmem/file_outerr"), name);
  }

  private static class FlushingFileRecordingOutputStream extends FileRecordingOutputStream {
    protected FlushingFileRecordingOutputStream(Path outputFile) {
      super(outputFile);
    }

    @Override
    public synchronized void write(byte[] b) throws IOException {
      super.write(b);
      flush();
    }

    @Override
    public synchronized void write(byte[] b, int off, int len) {
      super.write(b, off, len);
      try {
        flush();
      } catch (IOException e) {
        recordError(e);
      }
    }

    @Override
    public synchronized void write(int b) {
      super.write(b);
      try {
        flush();
      } catch (IOException e) {
        recordError(e);
      }
    }
  }

  public String getRecordedOutput() {
    return outAsLatin1() + errAsLatin1();
  }
}
