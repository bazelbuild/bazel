// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import java.io.IOException;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * A {@link SyscallCache} that delegates to a caching implementation only for paths with a
 * particular {@link FileSystem}.
 *
 * <p>Any calls that pass a {@link Path} backed by a different {@link FileSystem} are routed to
 * {@link SyscallCache#NO_CACHE}. This can be used to ensure that only calls for the build's main
 * {@link FileSystem} are cached. Common alternative filesystems for which caching is wasteful
 * include:
 *
 * <ul>
 *   <li>{@link
 *       com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.BundledFileSystem}, an
 *       in-memory filesystem (no real filesystem ops to save).
 *   <li>An {@linkplain com.google.devtools.build.lib.vfs.OutputService.ActionFileSystemType
 *       action-scoped filesystem} where there is no potential for reuse outside a single action's
 *       execution, and caching prolongs the lifetime of the instance.
 */
public final class SingleFileSystemSyscallCache implements SyscallCache {

  private final SyscallCache delegate;
  private final FileSystem fs;

  public SingleFileSystemSyscallCache(SyscallCache delegate, FileSystem fs) {
    this.delegate = checkNotNull(delegate);
    this.fs = checkNotNull(fs);
  }

  @Override
  public Collection<Dirent> readdir(Path path) throws IOException {
    return delegateFor(path).readdir(path);
  }

  @Nullable
  @Override
  public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
    return delegateFor(path).statIfFound(path, symlinks);
  }

  @Nullable
  @Override
  public DirentTypeWithSkip getType(Path path, Symlinks symlinks) throws IOException {
    return delegateFor(path).getType(path, symlinks);
  }

  @Override
  public byte[] getFastDigest(Path path) throws IOException {
    return delegateFor(path).getFastDigest(path);
  }

  @Override
  public byte[] getxattr(Path path, String xattrName) throws IOException {
    return delegateFor(path).getxattr(path, xattrName);
  }

  @Override
  public byte[] getxattr(Path path, String xattrName, Symlinks followSymlinks) throws IOException {
    return delegateFor(path).getxattr(path, xattrName, followSymlinks);
  }

  @Override
  public void noteAnalysisPhaseEnded() {
    delegate.noteAnalysisPhaseEnded();
  }

  @Override
  public void clear() {
    delegate.clear();
  }

  private SyscallCache delegateFor(Path path) {
    return path.getFileSystem().equals(fs) ? delegate : SyscallCache.NO_CACHE;
  }
}
