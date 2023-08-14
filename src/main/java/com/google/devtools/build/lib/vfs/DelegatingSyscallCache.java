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

/** {@link SyscallCache} that delegates to an injectable one. */
public class DelegatingSyscallCache implements SyscallCache {
  private SyscallCache delegate = SyscallCache.NO_CACHE;

  public void setDelegate(SyscallCache syscallCache) {
    this.delegate = checkNotNull(syscallCache);
  }

  @Override
  public Collection<Dirent> readdir(Path path) throws IOException {
    return delegate.readdir(path);
  }

  @Nullable
  @Override
  public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
    return delegate.statIfFound(path, symlinks);
  }

  @Nullable
  @Override
  public DirentTypeWithSkip getType(Path path, Symlinks symlinks) throws IOException {
    return delegate.getType(path, symlinks);
  }

  @Override
  public byte[] getFastDigest(Path path) throws IOException {
    return delegate.getFastDigest(path);
  }

  @Override
  public byte[] getxattr(Path path, String xattrName) throws IOException {
    return delegate.getxattr(path, xattrName);
  }

  @Override
  public byte[] getxattr(Path path, String xattrName, Symlinks followSymlinks) throws IOException {
    return delegate.getxattr(path, xattrName, followSymlinks);
  }

  @Override
  public void noteAnalysisPhaseEnded() {
    delegate.noteAnalysisPhaseEnded();
  }

  @Override
  public void clear() {
    delegate.clear();
  }
}
