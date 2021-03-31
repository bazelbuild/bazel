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

import com.google.common.base.Preconditions;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.ReadableByteChannel;
import java.util.Collection;

/**
 * A file system that delegates all operations to {@code delegateFs} under the hood.
 *
 * <p>This is helpful when wanting to implement a file system on top of another file system as it
 * allows code patterns like: <code>
 * {@literal @}Override
 * protected long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
 *   if (!someCondition) {
 *     return super.getFileSize(path, followSymlinks);
 *   }
 *   return rpc.getFileSize(path, followSymlinks);
 * }
 * </code>
 */
public abstract class DelegateFileSystem extends FileSystem {

  protected final FileSystem delegateFs;

  public DelegateFileSystem(FileSystem delegateFs) {
    super(Preconditions.checkNotNull(delegateFs, "delegateFs").getDigestFunction());
    this.delegateFs = delegateFs;
  }

  @Override
  public boolean supportsModifications(PathFragment path) {
    return delegateFs.supportsModifications(path);
  }

  @Override
  public boolean supportsSymbolicLinksNatively(PathFragment path) {
    return delegateFs.supportsSymbolicLinksNatively(path);
  }

  @Override
  protected boolean supportsHardLinksNatively(PathFragment path) {
    return delegateFs.supportsHardLinksNatively(path);
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return delegateFs.isFilePathCaseSensitive();
  }

  @Override
  public boolean createDirectory(PathFragment path) throws IOException {
    return delegateFs.createDirectory(path);
  }

  @Override
  public void createDirectoryAndParents(PathFragment path) throws IOException {
    delegateFs.createDirectoryAndParents(path);
  }

  @Override
  protected long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
    return delegateFs.getFileSize(path, followSymlinks);
  }

  @Override
  protected boolean delete(PathFragment path) throws IOException {
    return delegateFs.delete(path);
  }

  @Override
  protected long getLastModifiedTime(PathFragment path, boolean followSymlinks) throws IOException {
    return delegateFs.getLastModifiedTime(path, followSymlinks);
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    delegateFs.setLastModifiedTime(path, newTime);
  }

  @Override
  protected boolean isSymbolicLink(PathFragment path) {
    return delegateFs.isSymbolicLink(path);
  }

  @Override
  protected boolean isDirectory(PathFragment path, boolean followSymlinks) {
    return delegateFs.isDirectory(path, followSymlinks);
  }

  @Override
  protected boolean isFile(PathFragment path, boolean followSymlinks) {
    return delegateFs.isFile(path, followSymlinks);
  }

  @Override
  protected boolean isSpecialFile(PathFragment path, boolean followSymlinks) {
    return delegateFs.isSpecialFile(path, followSymlinks);
  }

  @Override
  protected void createSymbolicLink(PathFragment linkPath, PathFragment targetFragment)
      throws IOException {
    delegateFs.createSymbolicLink(linkPath, targetFragment);
  }

  @Override
  protected PathFragment readSymbolicLink(PathFragment path) throws IOException {
    return delegateFs.readSymbolicLink(path);
  }

  @Override
  protected boolean exists(PathFragment path, boolean followSymlinks) {
    return delegateFs.exists(path, followSymlinks);
  }

  @Override
  public boolean exists(PathFragment path) {
    return delegateFs.exists(path);
  }

  @Override
  protected Collection<String> getDirectoryEntries(PathFragment path) throws IOException {
    return delegateFs.getDirectoryEntries(path);
  }

  @Override
  protected boolean isReadable(PathFragment path) throws IOException {
    return delegateFs.isReadable(path);
  }

  @Override
  protected void setReadable(PathFragment path, boolean readable) throws IOException {
    delegateFs.setReadable(path, readable);
  }

  @Override
  protected boolean isWritable(PathFragment path) throws IOException {
    return delegateFs.isWritable(path);
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    delegateFs.setWritable(path, writable);
  }

  @Override
  protected boolean isExecutable(PathFragment path) throws IOException {
    return delegateFs.isExecutable(path);
  }

  @Override
  protected void setExecutable(PathFragment path, boolean executable) throws IOException {
    delegateFs.setExecutable(path, executable);
  }

  @Override
  protected InputStream getInputStream(PathFragment path) throws IOException {
    return delegateFs.getInputStream(path);
  }

  @Override
  protected ReadableByteChannel createReadableByteChannel(PathFragment path) throws IOException {
    return delegateFs.createReadableByteChannel(path);
  }

  @Override
  protected OutputStream getOutputStream(PathFragment path, boolean append) throws IOException {
    return delegateFs.getOutputStream(path, append);
  }

  @Override
  public void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    delegateFs.renameTo(sourcePath, targetPath);
  }

  @Override
  protected void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    delegateFs.createFSDependentHardLink(linkPath, originalPath);
  }

  @Override
  public String getFileSystemType(PathFragment path) {
    return delegateFs.getFileSystemType(path);
  }

  @Override
  protected void deleteTree(PathFragment path) throws IOException {
    delegateFs.deleteTree(path);
  }

  @Override
  protected void deleteTreesBelow(PathFragment dir) throws IOException {
    delegateFs.deleteTreesBelow(dir);
  }

  @Override
  public byte[] getxattr(PathFragment path, String name, boolean followSymlinks)
      throws IOException {
    return delegateFs.getxattr(path, name, followSymlinks);
  }

  @Override
  protected byte[] getFastDigest(PathFragment path) throws IOException {
    return delegateFs.getFastDigest(path);
  }

  @Override
  protected byte[] getDigest(PathFragment path) throws IOException {
    return delegateFs.getDigest(path);
  }

  @Override
  protected PathFragment resolveOneLink(PathFragment path) throws IOException {
    return delegateFs.resolveOneLink(path);
  }

  @Override
  protected Path resolveSymbolicLinks(PathFragment path) throws IOException {
    return getPath(delegateFs.resolveSymbolicLinks(path).asFragment());
  }

  @Override
  protected FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    return delegateFs.stat(path, followSymlinks);
  }

  @Override
  protected FileStatus statNullable(PathFragment path, boolean followSymlinks) {
    return delegateFs.statNullable(path, followSymlinks);
  }

  @Override
  protected FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
    return delegateFs.statIfFound(path, followSymlinks);
  }

  @Override
  protected PathFragment readSymbolicLinkUnchecked(PathFragment path) throws IOException {
    return delegateFs.readSymbolicLink(path);
  }

  @Override
  protected Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
      throws IOException {
    return delegateFs.readdir(path, followSymlinks);
  }

  @Override
  protected void chmod(PathFragment path, int mode) throws IOException {
    delegateFs.chmod(path, mode);
  }

  @Override
  protected void createHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    delegateFs.createHardLink(linkPath, originalPath);
  }

  @Override
  protected void prefetchPackageAsync(PathFragment path, int maxDirs) {
    delegateFs.prefetchPackageAsync(path, maxDirs);
  }
}
