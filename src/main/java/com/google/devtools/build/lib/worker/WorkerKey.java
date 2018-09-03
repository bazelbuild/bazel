// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;

/**
 * Data container that uniquely identifies a kind of worker process and is used as the key for the
 * {@link WorkerPool}.
 */
final class WorkerKey {
  private final ImmutableList<String> args;
  private final ImmutableMap<String, String> env;
  private final Path execRoot;
  private final String mnemonic;

  /**
   * These are used during validation whether a worker is still usable. They are not used to
   * uniquely identify a kind of worker, thus it is not to be used by the .equals() / .hashCode()
   * methods.
   */
  private final HashCode workerFilesCombinedHash;
  private final SortedMap<PathFragment, HashCode> workerFilesWithHashes;
  private final boolean mustBeSandboxed;

  WorkerKey(
      List<String> args,
      Map<String, String> env,
      Path execRoot,
      String mnemonic,
      HashCode workerFilesCombinedHash,
      SortedMap<PathFragment, HashCode> workerFilesWithHashes,
      boolean mustBeSandboxed) {
    this.args = ImmutableList.copyOf(Preconditions.checkNotNull(args));
    this.env = ImmutableMap.copyOf(Preconditions.checkNotNull(env));
    this.execRoot = Preconditions.checkNotNull(execRoot);
    this.mnemonic = Preconditions.checkNotNull(mnemonic);
    this.workerFilesCombinedHash = Preconditions.checkNotNull(workerFilesCombinedHash);
    this.workerFilesWithHashes = Preconditions.checkNotNull(workerFilesWithHashes);
    this.mustBeSandboxed = mustBeSandboxed;
  }

  public ImmutableList<String> getArgs() {
    return args;
  }

  public ImmutableMap<String, String> getEnv() {
    return env;
  }

  public Path getExecRoot() {
    return execRoot;
  }

  public String getMnemonic() {
    return mnemonic;
  }

  public HashCode getWorkerFilesCombinedHash() {
    return workerFilesCombinedHash;
  }

  public SortedMap<PathFragment, HashCode> getWorkerFilesWithHashes() {
    return workerFilesWithHashes;
  }

  public boolean mustBeSandboxed() {
    return mustBeSandboxed;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    WorkerKey workerKey = (WorkerKey) o;

    if (!args.equals(workerKey.args)) {
      return false;
    }
    if (!env.equals(workerKey.env)) {
      return false;
    }
    if (!execRoot.equals(workerKey.execRoot)) {
      return false;
    }
    return mnemonic.equals(workerKey.mnemonic);

  }

  @Override
  public int hashCode() {
    int result = args.hashCode();
    result = 31 * result + env.hashCode();
    result = 31 * result + execRoot.hashCode();
    result = 31 * result + mnemonic.hashCode();
    return result;
  }

  @Override
  public String toString() {
    return Spawns.asShellCommand(args, execRoot, env, /* prettyPrintArgs= */ false);
  }
}
