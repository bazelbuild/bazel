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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DefaultHashFunctionNotSetException;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** This class implements the FileSystem interface using direct calls to the UNIX filesystem. */
@ThreadSafe
public abstract class AbstractFileSystem extends FileSystem {

  protected static final String ERR_PERMISSION_DENIED = " (Permission denied)";
  protected static final Profiler profiler = Profiler.instance();

  public AbstractFileSystem() throws DefaultHashFunctionNotSetException {}

  public AbstractFileSystem(DigestHashFunction digestFunction) {
    super(digestFunction);
  }

  @Override
  protected InputStream getInputStream(Path path) throws IOException {
    // This loop is a workaround for an apparent bug in FileInputStream.open, which delegates
    // ultimately to JVM_Open in the Hotspot JVM.  This call is not EINTR-safe, so we must do the
    // retry here.
    for (; ; ) {
      try {
        return createFileInputStream(path);
      } catch (FileNotFoundException e) {
        if (e.getMessage().endsWith("(Interrupted system call)")) {
          continue;
        } else {
          throw e;
        }
      }
    }
  }

  /** Returns either normal or profiled FileInputStream. */
  private InputStream createFileInputStream(Path path) throws FileNotFoundException {
    final String name = path.toString();
    if (profiler.isActive()
        && (profiler.isProfiling(ProfilerTask.VFS_READ)
            || profiler.isProfiling(ProfilerTask.VFS_OPEN))) {
      long startTime = Profiler.nanoTimeMaybe();
      try {
        // Replace default FileInputStream instance with the custom one that does profiling.
        return new ProfiledFileInputStream(name);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_OPEN, name);
      }
    } else {
      // Use normal FileInputStream instance if profiler is not enabled.
      return new FileInputStream(path.toString());
    }
  }

  /**
   * Returns either normal or profiled FileOutputStream. Should be used by subclasses to create
   * default OutputStream instance.
   */
  protected OutputStream createFileOutputStream(Path path, boolean append)
      throws FileNotFoundException {
    final String name = path.toString();
    if (profiler.isActive()
        && (profiler.isProfiling(ProfilerTask.VFS_WRITE)
            || profiler.isProfiling(ProfilerTask.VFS_OPEN))) {
      long startTime = Profiler.nanoTimeMaybe();
      try {
        return new ProfiledFileOutputStream(name, append);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_OPEN, name);
      }
    } else {
      return new FileOutputStream(name, append);
    }
  }

  @Override
  protected OutputStream getOutputStream(Path path, boolean append) throws IOException {
    try {
      return createFileOutputStream(path, append);
    } catch (FileNotFoundException e) {
      // Why does it throw a *FileNotFoundException* if it can't write?
      // That does not make any sense! And its in a completely different
      // format than in other situations, no less!
      if (e.getMessage().equals(path + ERR_PERMISSION_DENIED)) {
        throw new FileAccessException(e.getMessage());
      }
      throw e;
    }
  }

  private static final class ProfiledFileInputStream extends FileInputStream {
    private final String name;

    public ProfiledFileInputStream(String name) throws FileNotFoundException {
      super(name);
      this.name = name;
    }

    @Override
    public int read() throws IOException {
      long startTime = Profiler.nanoTimeMaybe();
      try {
        // Note that FileInputStream#read() does *not* call any of our overridden methods,
        // so there's no concern with double counting here.
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
      long startTime = Profiler.nanoTimeMaybe();
      try {
        return super.read(b, off, len);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_READ, name);
      }
    }
  }

  private static final class ProfiledFileOutputStream extends FileOutputStream {
    private final String name;

    public ProfiledFileOutputStream(String name, boolean append) throws FileNotFoundException {
      super(name, append);
      this.name = name;
    }

    @Override
    public void write(byte[] b) throws IOException {
      write(b, 0, b.length);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      long startTime = Profiler.nanoTimeMaybe();
      try {
        super.write(b, off, len);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_WRITE, name);
      }
    }
  }
}
