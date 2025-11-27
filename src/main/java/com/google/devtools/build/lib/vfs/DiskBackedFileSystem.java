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
package com.google.devtools.build.lib.vfs;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;

/**
 * This class extends {@link FileSystem} with default implementations providing access to files on
 * disk through standard library APIs.
 */
@ThreadSafe
public abstract class DiskBackedFileSystem extends FileSystem {
  private static final Profiler profiler = Profiler.instance();

  private static final ImmutableSet<StandardOpenOption> READ_WRITE_BYTE_CHANNEL_OPEN_OPTIONS =
      ImmutableSet.of(
          StandardOpenOption.READ,
          StandardOpenOption.WRITE,
          StandardOpenOption.CREATE,
          StandardOpenOption.TRUNCATE_EXISTING);

  protected DiskBackedFileSystem(DigestHashFunction hashFunction) {
    super(hashFunction);
  }

  // Force subclasses to override getIoFile and getNioPath, as the methods below require them.

  @Override
  public abstract File getIoFile(PathFragment path);

  @Override
  public abstract java.nio.file.Path getNioPath(PathFragment path);

  @Override
  public InputStream getInputStream(PathFragment path) throws IOException {
    File file = checkNotNull(getIoFile(path), "getIoFile() must not be null");

    boolean profileOpen = profiler.isActive() && profiler.isProfiling(ProfilerTask.VFS_OPEN);
    boolean profileRead = profiler.isActive() && profiler.isProfiling(ProfilerTask.VFS_READ);

    long startTime = profiler.nanoTimeMaybe();
    try {
      return profileRead
          ? new ProfiledFileInputStream(file, path.getPathString())
          : new FileInputStream(file);
    } catch (FileNotFoundException e) {
      // FileInputStream throws FileNotFoundException if opening fails for any reason, including
      // permissions. Fix it up here.
      if (e.getMessage().endsWith(ERR_PERMISSION_DENIED)) {
        throw new FileAccessException(e.getMessage());
      }
      throw e;
    } finally {
      if (profileOpen) {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_OPEN, path.getPathString());
      }
    }
  }

  @Override
  public OutputStream getOutputStream(PathFragment path, boolean append, boolean internal)
      throws IOException {
    File file = checkNotNull(getIoFile(path), "getIoFile() must not be null");

    boolean profileOpen =
        !internal && profiler.isActive() && profiler.isProfiling(ProfilerTask.VFS_OPEN);
    boolean profileWrite =
        !internal && profiler.isActive() && profiler.isProfiling(ProfilerTask.VFS_WRITE);

    long startTime = profiler.nanoTimeMaybe();
    try {
      return profileWrite
          ? new ProfiledFileOutputStream(file, append, path.getPathString())
          : new FileOutputStream(file, append);
    } catch (FileNotFoundException e) {
      // FileOutputStream throws FileNotFoundException if opening fails for any reason, including
      // permissions. Fix it up here.
      if (e.getMessage().endsWith(ERR_PERMISSION_DENIED)) {
        throw new FileAccessException(e.getMessage());
      }
      throw e;
    } finally {
      if (profileOpen) {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_OPEN, path.getPathString());
      }
    }
  }

  @Override
  public SeekableByteChannel createReadWriteByteChannel(PathFragment path) throws IOException {
    java.nio.file.Path nioPath = checkNotNull(getNioPath(path), "getNioPath() must not be null");

    boolean profileOpen = profiler.isActive() && profiler.isProfiling(ProfilerTask.VFS_OPEN);

    long startTime = Profiler.instance().nanoTimeMaybe();
    try {
      // TODO: add profiling for read/write operations.
      return Files.newByteChannel(nioPath, READ_WRITE_BYTE_CHANNEL_OPEN_OPTIONS);
    } finally {
      if (profileOpen) {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_OPEN, path.toString());
      }
    }
  }

  /**
   * A {@link FileInputStream} that adds profile traces around read operations.
   *
   * <p>Implementation note: this class extends {@link FileInputStream} instead of wrapping around
   * it so that {@code instanceof FileInputStream} checks still work.
   */
  private static class ProfiledFileInputStream extends FileInputStream {
    private final String name;

    private ProfiledFileInputStream(File file, String name) throws IOException {
      super(file);
      this.name = name;
    }

    @Override
    public int read() throws IOException {
      long startTime = profiler.nanoTimeMaybe();
      try {
        return super.read();
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_READ, name);
      }
    }

    @Override
    public int read(byte[] b) throws IOException {
      return read(b, 0, b.length);
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
      long startTime = profiler.nanoTimeMaybe();
      try {
        return super.read(b, off, len);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_READ, name);
      }
    }
  }

  /**
   * A {@link FileOutputStream} that adds profile traces around write operations.
   *
   * <p>Implementation note: this class extends {@link FileOutputStream} instead of wrapping around
   * it so that {@code instanceof FileOutputStream} checks still work.
   */
  private static class ProfiledFileOutputStream extends FileOutputStream {
    private final String name;

    private ProfiledFileOutputStream(File file, boolean append, String name) throws IOException {
      super(file, append);
      this.name = name;
    }

    @Override
    public void write(int b) throws IOException {
      long startTime = profiler.nanoTimeMaybe();
      try {
        super.write(b);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_WRITE, name);
      }
    }

    @Override
    public void write(byte[] b) throws IOException {
      write(b, 0, b.length);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      long startTime = profiler.nanoTimeMaybe();
      try {
        super.write(b, off, len);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_WRITE, name);
      }
    }
  }
}
