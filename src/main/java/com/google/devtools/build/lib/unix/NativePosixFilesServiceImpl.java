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

package com.google.devtools.build.lib.unix;

import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.jni.JniLoader;
import java.io.FileDescriptor;
import javax.annotation.Nullable;

/** Implementation of {@link NativePosixFilesService}. */
public final class NativePosixFilesServiceImpl implements NativePosixFilesService {

  public NativePosixFilesServiceImpl() {}

  static {
    JniLoader.loadJni();
  }

  @Override
  @Nullable
  public native String readlink(String path) throws NativePosixFilesException;

  @Override
  public native void chmod(String path, int mode) throws NativePosixFilesException;

  @Override
  public native void symlink(String oldpath, String newpath) throws NativePosixFilesException;

  @Override
  public native void link(String oldpath, String newpath) throws NativePosixFilesException;

  @Override
  public Stat stat(String path, StatErrorHandling errorHandling) throws NativePosixFilesException {
    return stat(path, errorHandling.getCode());
  }

  private native Stat stat(String path, char errorHandling) throws NativePosixFilesException;

  @Override
  public Stat lstat(String path, StatErrorHandling errorHandling) throws NativePosixFilesException {
    return lstat(path, errorHandling.getCode());
  }

  private native Stat lstat(String path, char errorHandling) throws NativePosixFilesException;

  @Override
  public native void utimensat(String path, boolean now, long epochMilli)
      throws NativePosixFilesException;

  @Override
  public native boolean mkdir(String path, int mode) throws NativePosixFilesException;

  @Override
  public native Dirent[] readdir(String path) throws NativePosixFilesException;

  @Override
  public native void rename(String oldpath, String newpath) throws NativePosixFilesException;

  @Override
  public native boolean remove(String path) throws NativePosixFilesException;

  @Override
  public native void mkfifo(String path, int mode) throws NativePosixFilesException;

  @Override
  public native int drainFifoNonBlocking(FileDescriptor fd) throws NativePosixFilesException;

  @Override
  public native byte[] getxattr(String path, String name) throws NativePosixFilesException;

  @Override
  public native byte[] lgetxattr(String path, String name) throws NativePosixFilesException;

  @Override
  public native void deleteTreesBelow(String dir) throws NativePosixFilesException;

  /** Logs a path string that does not have a Latin-1 coder. Called from JNI. */
  private static void logBadPath(String path) {
    BugReport.sendNonFatalBugReport(
        new IllegalStateException("Path string does not have a Latin-1 coder: %s".formatted(path)));
  }

  /** Concrete implementation of {@link NativePosixFilesService.Dirent} as a record. */
  public static record DirentImpl(String name, NativePosixFilesService.Dirent.Type type)
      implements NativePosixFilesService.Dirent {}

  /** Concrete implementation of {@link NativePosixFilesService.Stat} as a record. */
  @SuppressWarnings("GoodTime")
  public static record StatImpl(int mode, long mtime, long ctime, long size, long ino)
      implements NativePosixFilesService.Stat {}
}
