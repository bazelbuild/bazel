// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.vfs;

import com.google.common.base.Preconditions;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;

/**
 * A file system that delegates all operations to {@code delegateFs} under the hood.
 *
 * <p>This is helpful when wanting to implement a file system on top of another file
 * system as it allows code patterns like:
 *
 * <code>
 * @Override
 * protected long getFileSize(Path path, boolean followSymlinks) throws IOException {
 *   if (!someCondition) {
 *     return super.getFileSize(path, followSymlinks);
 *   }
 *   return rpc.getFileSize(path, followSymlinks);
 * }
 * </code>
 */
public class DelegateFileSystem extends FileSystem {

  private final FileSystem delegateFs;

  public DelegateFileSystem(FileSystem delegateFs) {
    super(delegateFs.getDigestFunction());
    this.delegateFs = Preconditions.checkNotNull(delegateFs, "delegateFs");
  }

  @Override
  public boolean supportsModifications(Path path) {
    return delegateFs.supportsModifications(toDelegatePath(path));
  }

  @Override
  public boolean supportsSymbolicLinksNatively(Path path) {
    return delegateFs.supportsSymbolicLinksNatively(toDelegatePath(path));
  }

  @Override
  protected boolean supportsHardLinksNatively(Path path) {
    return delegateFs.supportsHardLinksNatively(toDelegatePath(path));
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return delegateFs.isFilePathCaseSensitive();
  }

  @Override
  public boolean createDirectory(Path path) throws IOException {
    return delegateFs.createDirectory(toDelegatePath(path));
  }

  @Override
  public void createDirectoryAndParents(Path path) throws IOException {
    delegateFs.createDirectoryAndParents(toDelegatePath(path));
  }

  @Override
  protected long getFileSize(Path path, boolean followSymlinks) throws IOException {
    return delegateFs.getFileSize(toDelegatePath(path), followSymlinks);
  }

  @Override
  public boolean delete(Path path) throws IOException {
    return delegateFs.delete(toDelegatePath(path));
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks) throws IOException {
    return delegateFs.getLastModifiedTime(toDelegatePath(path), followSymlinks);
  }

  @Override
  public void setLastModifiedTime(Path path, long newTime) throws IOException {
    delegateFs.setLastModifiedTime(toDelegatePath(path), newTime);
  }

  @Override
  protected boolean isSymbolicLink(Path path) {
    return delegateFs.isSymbolicLink(toDelegatePath(path));
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    return delegateFs.isDirectory(toDelegatePath(path), followSymlinks);
  }

  @Override
  protected boolean isFile(Path path, boolean followSymlinks) {
    return delegateFs.isFile(toDelegatePath(path), followSymlinks);
  }

  @Override
  protected boolean isSpecialFile(Path path, boolean followSymlinks) {
    return delegateFs.isSpecialFile(toDelegatePath(path), followSymlinks);
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) throws IOException {
    delegateFs.createSymbolicLink(toDelegatePath(linkPath), targetFragment);
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    return delegateFs.readSymbolicLink(toDelegatePath(path));
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    return delegateFs.exists(toDelegatePath(path), followSymlinks);
  }

  @Override
  protected Collection<String> getDirectoryEntries(Path path) throws IOException {
    return delegateFs.getDirectoryEntries(toDelegatePath(path));
  }

  @Override
  protected boolean isReadable(Path path) throws IOException {
    return delegateFs.isReadable(toDelegatePath(path));
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {
    delegateFs.setReadable(toDelegatePath(path), readable);
  }

  @Override
  protected boolean isWritable(Path path) throws IOException {
    return delegateFs.isWritable(toDelegatePath(path));
  }

  @Override
  public void setWritable(Path path, boolean writable) throws IOException {
    delegateFs.setWritable(toDelegatePath(path), writable);
  }

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    return delegateFs.isExecutable(toDelegatePath(path));
  }

  @Override
  protected void setExecutable(Path path, boolean executable) throws IOException {
    delegateFs.setExecutable(toDelegatePath(path), executable);
  }

  @Override
  protected InputStream getInputStream(Path path) throws IOException {
    return delegateFs.getInputStream(toDelegatePath(path));
  }

  @Override
  protected OutputStream getOutputStream(Path path, boolean append) throws IOException {
    return delegateFs.getOutputStream(toDelegatePath(path), append);
  }

  @Override
  public void renameTo(Path sourcePath, Path targetPath) throws IOException {
    delegateFs.renameTo(toDelegatePath(sourcePath), toDelegatePath(targetPath));
  }

  @Override
  protected void createFSDependentHardLink(Path linkPath, Path originalPath) throws IOException {
    delegateFs.createFSDependentHardLink(toDelegatePath(linkPath), toDelegatePath(originalPath));
  }

  @Override
  public String getFileSystemType(Path path) {
    return delegateFs.getFileSystemType(toDelegatePath(path));
  }

  @Override
  public void deleteTree(Path path) throws IOException {
    delegateFs.deleteTree(toDelegatePath(path));
  }

  @Override
  public void deleteTreesBelow(Path dir) throws IOException {
    delegateFs.deleteTreesBelow(toDelegatePath(dir));
  }

  @Override
  public byte[] getxattr(Path path, String name, boolean followSymlinks) throws IOException {
    return delegateFs.getxattr(toDelegatePath(path), name, followSymlinks);
  }

  @Override
  protected byte[] getFastDigest(Path path) throws IOException {
    return delegateFs.getFastDigest(toDelegatePath(path));
  }

  @Override
  protected byte[] getDigest(Path path) throws IOException {
    return delegateFs.getDigest(toDelegatePath(path));
  }

  @Override
  protected PathFragment resolveOneLink(Path path) throws IOException {
    return delegateFs.resolveOneLink(toDelegatePath(path));
  }

  @Override
  protected Path resolveSymbolicLinks(Path path) throws IOException {
    return fromDelegatePath(delegateFs.resolveSymbolicLinks(toDelegatePath(path)));
  }

  @Override
  protected FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    return delegateFs.stat(toDelegatePath(path), followSymlinks);
  }

  @Override
  protected FileStatus statNullable(Path path, boolean followSymlinks) {
    return delegateFs.statNullable(toDelegatePath(path), followSymlinks);
  }

  @Override
  protected FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
    return delegateFs.statIfFound(toDelegatePath(path), followSymlinks);
  }

  @Override
  protected PathFragment readSymbolicLinkUnchecked(Path path) throws IOException {
    return delegateFs.readSymbolicLink(toDelegatePath(path));
  }

  @Override
  public boolean exists(Path path) {
    return delegateFs.exists(toDelegatePath(path));
  }

  @Override
  protected Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
    return delegateFs.readdir(toDelegatePath(path), followSymlinks);
  }

  @Override
  protected void chmod(Path path, int mode) throws IOException {
    delegateFs.chmod(toDelegatePath(path), mode);
  }

  @Override
  protected void createHardLink(Path linkPath, Path originalPath) throws IOException {
    delegateFs.createHardLink(toDelegatePath(linkPath), toDelegatePath(originalPath));
  }

  @Override
  protected void prefetchPackageAsync(Path path, int maxDirs) {
    delegateFs.prefetchPackageAsync(toDelegatePath(path), maxDirs);
  }

  protected final Path toDelegatePath(Path path) {
    Preconditions.checkArgument(path.getFileSystem() == this);
    return Path.create(path.getPathString(), delegateFs);
  }

  protected final Path fromDelegatePath(Path delegatePath) {
    Preconditions.checkArgument(delegatePath.getFileSystem() == delegateFs);
    return Path.create(delegatePath.getPathString(), this);
  }
}
