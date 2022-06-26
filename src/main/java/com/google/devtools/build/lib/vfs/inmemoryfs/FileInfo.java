// Copyright 2019 The Bazel Authors. All rights reserved.
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
//
package com.google.devtools.build.lib.vfs.inmemoryfs;

import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.ReadableByteChannel;

/**
 * This interface represents a mutable file stored in an InMemoryFileSystem.
 */
@ThreadSafe
public abstract class FileInfo extends InMemoryContentInfo {

  protected FileInfo(Clock clock) {
    super(clock);
  }

  @Override
  public boolean isDirectory() {
    return false;
  }

  @Override
  public boolean isSymbolicLink() {
    return false;
  }

  @Override
  public boolean isFile() {
    return true;
  }

  @Override
  public boolean isSpecialFile() {
    return false;
  }

  public abstract OutputStream getOutputStream(boolean append) throws IOException;

  public abstract InputStream getInputStream() throws IOException;

  public ReadableByteChannel createReadableByteChannel() {
    throw new UnsupportedOperationException();
  }

  public abstract byte[] getxattr(String name) throws IOException;

  public abstract byte[] getFastDigest() throws IOException;
}
