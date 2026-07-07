// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStatusWithMetadata;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.FileInfo;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryContentInfo;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.SeekableByteChannel;

/**
 * An in-memory file system that stores the metadata (but not the contents) of remotely stored
 * files.
 */
class RemoteInMemoryFileSystem extends InMemoryFileSystem {

  RemoteInMemoryFileSystem(DigestHashFunction hashFunction) {
    super(hashFunction);
  }

  @Override
  public synchronized OutputStream getOutputStream(
      PathFragment path, boolean append, boolean internal) throws IOException {
    // To get an output stream from remote file, we need to first stage it.
    throw new IllegalStateException("Shouldn't be called directly");
  }

  @Override
  protected FileInfo newFile(Clock clock, PathFragment path) {
    return new RemoteInMemoryFileInfo(clock);
  }

  protected void injectFile(PathFragment path, FileArtifactValue metadata) throws IOException {
    checkArgument(metadata.isRemote(), "metadata is not remote: %s", metadata);
    createDirectoryAndParents(path.getParentDirectory());
    InMemoryContentInfo node = getOrCreateWritableInode(path);
    // If a node already exists but is not a regular file, throw an error.
    if (!(node instanceof RemoteInMemoryFileInfo remoteInMemoryFileInfo)) {
      throw new IOException("Could not inject into " + node);
    }

    remoteInMemoryFileInfo.set(metadata);
  }

  static class RemoteInMemoryFileInfo extends FileInfo implements FileStatusWithMetadata {
    private FileArtifactValue metadata;

    RemoteInMemoryFileInfo(Clock clock) {
      super(clock);
    }

    private void set(FileArtifactValue metadata) {
      this.metadata = metadata;
    }

    @Override
    public OutputStream getOutputStream(boolean append) throws IOException {
      throw new IllegalStateException("Shouldn't be called directly");
    }

    @Override
    public InputStream getInputStream() throws IOException {
      throw new IllegalStateException("Shouldn't be called directly");
    }

    @Override
    public SeekableByteChannel createReadWriteByteChannel() throws IOException {
      throw new IllegalStateException("Shouldn't be called directly");
    }

    @Override
    public byte[] getxattr(String name) throws IOException {
      throw new IllegalStateException("Shouldn't be called directly");
    }

    @Override
    public byte[] getFastDigest() {
      return metadata.getDigest();
    }

    @Override
    public byte[] getDigest() throws IOException {
      return metadata.getDigest();
    }

    @Override
    public long getSize() {
      return metadata.getSize();
    }

    @Override
    public FileArtifactValue getMetadata() {
      return metadata;
    }
  }
}
