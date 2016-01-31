// Copyright 2016 The Bazel Authors. All rights reserved.
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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.attribute.BasicFileAttributes;

/**
 * Jury-rigged file system for Windows.
 */
@ThreadSafe
public class WindowsFileSystem extends JavaIoFileSystem {

  public static final LinkOption[] NO_OPTIONS = new LinkOption[0];
  public static final LinkOption[] NO_FOLLOW = new LinkOption[]{LinkOption.NOFOLLOW_LINKS};

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment)
      throws IOException {
    // TODO(lberki): Add some JNI to create hard links/junctions instead of calling out to
    // cmd.exe
    File file = getIoFile(linkPath);
    try {
      File targetFile = new File(targetFragment.getPathString());
      if (targetFile.isDirectory()) {
        createDirectoryJunction(targetFile, file);
      } else {
        Files.copy(targetFile.toPath(), file.toPath());
      }
    } catch (java.nio.file.FileAlreadyExistsException e) {
      throw new IOException(linkPath + ERR_FILE_EXISTS);
    } catch (java.nio.file.AccessDeniedException e) {
      throw new IOException(linkPath + ERR_PERMISSION_DENIED);
    } catch (java.nio.file.NoSuchFileException e) {
      throw new FileNotFoundException(linkPath + ERR_NO_SUCH_FILE_OR_DIR);
    }
  }

  private void createDirectoryJunction(File sourceDirectory, File targetPath) throws IOException {
    StringBuilder cl = new StringBuilder("cmd.exe /c ");
    cl.append("mklink /J ");
    cl.append('"');
    cl.append(targetPath.getAbsolutePath());
    cl.append('"');
    cl.append(' ');
    cl.append('"');
    cl.append(sourceDirectory.getAbsolutePath());
    cl.append('"');
    Process process = Runtime.getRuntime().exec(cl.toString());
    try {
      process.waitFor();
      if (process.exitValue() != 0) {
        throw new IOException("Command failed " + cl);
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Command failed ", e);
    }
  }

  @Override
  protected boolean fileIsSymbolicLink(File file) {
    try {
      if (file.isDirectory() && isJunction(file.toPath())) {
        return true;
      }
    } catch (IOException e) {
      // Did not work, try in another way
    }
    return super.fileIsSymbolicLink(file);
  }

  private LinkOption[] linkOpts(boolean followSymlinks) {
    return followSymlinks ? NO_OPTIONS : NO_FOLLOW;
  }

  @Override
  protected FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    File file = getIoFile(path);
    final BasicFileAttributes attributes;
    try {
      attributes = Files.readAttributes(
          file.toPath(), BasicFileAttributes.class, linkOpts(followSymlinks));
    } catch (java.nio.file.FileSystemException e) {
      throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
    }

    final boolean isSymbolicLink = !followSymlinks && fileIsSymbolicLink(file);
    FileStatus status =  new FileStatus() {
      @Override
      public boolean isFile() {
        return attributes.isRegularFile() || (isSpecialFile() && !isDirectory());
      }

      @Override
      public boolean isSpecialFile() {
        return attributes.isOther();
      }

      @Override
      public boolean isDirectory() {
        return attributes.isDirectory();
      }

      @Override
      public boolean isSymbolicLink() {
        return isSymbolicLink;
      }

      @Override
      public long getSize() throws IOException {
        return attributes.size();
      }

      @Override
      public long getLastModifiedTime() throws IOException {
        return attributes.lastModifiedTime().toMillis();
      }

      @Override
      public long getLastChangeTime() {
        // This is the best we can do with Java NIO...
        return attributes.lastModifiedTime().toMillis();
      }

      @Override
      public long getNodeId() {
        // TODO(bazel-team): Consider making use of attributes.fileKey().
        return -1;
      }
    };

    return status;
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    if (!followSymlinks) {
      try {
        if (isJunction(getIoFile(path).toPath())) {
          return false;
        }
      } catch (IOException e) {
        e.printStackTrace();
        return false;
      }
    }
    return super.isDirectory(path, followSymlinks);
  }

  private static boolean isJunction(java.nio.file.Path p) throws IOException {
    // Jury-rigged
    return p.compareTo(p.toRealPath()) != 0;
  }
}
