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
  /** A WorkerProxy will be instantiated if true, instantiate a regular Worker if false. */
  private final boolean proxied;

  WorkerKey(
      ImmutableList<String> args,
      ImmutableMap<String, String> env,
      Path execRoot,
      String mnemonic,
      HashCode workerFilesCombinedHash,
      SortedMap<PathFragment, HashCode> workerFilesWithHashes,
      boolean mustBeSandboxed,
      boolean proxied) {
    /** Build options. */
    this.args = Preconditions.checkNotNull(args);
    /** Environment variables. */
    this.env = Preconditions.checkNotNull(env);
    /** Execution root of Bazel process. */
    this.execRoot = Preconditions.checkNotNull(execRoot);
    /** Mnemonic of the worker. */
    this.mnemonic = Preconditions.checkNotNull(mnemonic);
    /** One combined hash code for all files. */
    this.workerFilesCombinedHash = Preconditions.checkNotNull(workerFilesCombinedHash);
    /** Worker files with the corresponding hash code. */
    this.workerFilesWithHashes = Preconditions.checkNotNull(workerFilesWithHashes);
    /** Set it to true if this job should be run in sandbox. */
    this.mustBeSandboxed = mustBeSandboxed;
    /** Set it to true if this job should be run with WorkerProxy. */
    this.proxied = proxied;
  }

  /** Getter function for variable args. */
  public ImmutableList<String> getArgs() {
    return args;
  }

  /** Getter function for variable env. */
  public ImmutableMap<String, String> getEnv() {
    return env;
  }

  /** Getter function for variable execRoot. */
  public Path getExecRoot() {
    return execRoot;
  }

  /** Getter function for variable mnemonic. */
  public String getMnemonic() {
    return mnemonic;
  }

  /** Getter function for variable workerFilesCombinedHash. */
  public HashCode getWorkerFilesCombinedHash() {
    return workerFilesCombinedHash;
  }

  /** Getter function for variable workerFilesWithHashes. */
  public SortedMap<PathFragment, HashCode> getWorkerFilesWithHashes() {
    return workerFilesWithHashes;
  }

  /** Getter function for variable mustBeSandboxed. */
  public boolean mustBeSandboxed() {
    return mustBeSandboxed;
  }

  /** Getter function for variable proxied. */
  public boolean getProxied() {
    return proxied;
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
