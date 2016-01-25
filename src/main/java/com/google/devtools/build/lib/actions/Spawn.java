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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;

/**
 * An object representing a subprocess to be invoked, including its command and
 * arguments, its working directory, its environment, a boolean indicating
 * whether remote execution is appropriate for this command, and if so, the set
 * of files it is expected to read and write.
 */
public interface Spawn {

  /**
   * Returns true iff this command may be executed remotely.
   */
  boolean isRemotable();

  /**
   * Out-of-band data for this spawn. This can be used to signal hints (hardware requirements,
   * local vs. remote) to the execution subsystem.
   *
   * <p>String tags from {@link
   * com.google.devtools.build.lib.rules.test.TestTargetProperties#getExecutionInfo()} can be added
   * as keys with arbitrary values to this map too.
   */
  ImmutableMap<String, String> getExecutionInfo();

  /**
   * Returns this Spawn as a Bourne shell command.
   *
   * @param workingDir the initial working directory of the command
   */
  String asShellCommand(Path workingDir);

  /**
   * Returns the runfiles data for remote execution. Format is (directory, manifest file).
   */
  ImmutableMap<PathFragment, Artifact> getRunfilesManifests();

  /**
   * Returns the {@link RunfilesSupplier} helper encapsulating the runfiles for this spawn.
   */
  RunfilesSupplier getRunfilesSupplier();

  /**
   * Returns artifacts for filesets, so they can be scheduled on remote execution.
   */
  ImmutableList<Artifact> getFilesetManifests();

  /**
   * Returns a protocol buffer describing this spawn for use by the extra_action functionality.
   */
  SpawnInfo getExtraActionInfo();

  /**
   * Returns the command (the first element) and its arguments.
   */
  ImmutableList<String> getArguments();

  /**
   * Returns the initial environment of the process.
   * If null, the environment is inherited from the parent process.
   */
  ImmutableMap<String, String> getEnvironment();

  /**
   * Returns the list of files that are required to execute this spawn (e.g. the compiler binary),
   * in contrast to files necessary for the tool to do its work (e.g. source code to be compiled).
   *
   * <p>The returned set of files is a subset of what getInputFiles() returns.
   *
   * <p>This method explicitly does not expand middleman artifacts. Pass the result
   * to an appropriate utility method on {@link com.google.devtools.build.lib.actions.Artifact} to
   * expand the middlemen.
   *
   * <p>This is for use with persistent workers, so we can restart workers when their binaries
   * have changed.
   */
  Iterable<? extends ActionInput> getToolFiles();

  /**
   * Returns the list of files that this command may read.
   *
   * <p>This method explicitly does not expand middleman artifacts. Pass the result
   * to an appropriate utility method on {@link com.google.devtools.build.lib.actions.Artifact} to
   * expand the middlemen.
   *
   * <p>This is for use with remote execution, so we can ship inputs before starting the
   * command. Order stability across multiple calls should be upheld for performance reasons.
   */
  Iterable<? extends ActionInput> getInputFiles();

  /**
   * Returns the collection of files that this command must write.  Callers should not mutate
   * the result.
   *
   * <p>This is for use with remote execution, so remote execution does not have to guess what
   * outputs the process writes.  While the order does not affect the semantics, it should be
   * stable so it can be cached.
   */
  Collection<? extends ActionInput> getOutputFiles();

  /**
   * Instructs the spawn strategy to try to fetch these optional output files in addition to the
   * usual output artifacts. The PathFragments should be relative to the exec root.
   */
  Collection<PathFragment> getOptionalOutputFiles();

  /**
   * Returns the resource owner for local fallback.
   */
  ActionMetadata getResourceOwner();

  /**
   * Returns the amount of resources needed for local fallback.
   */
  ResourceSet getLocalResources();

  /**
   * Returns the owner for this action. Production code should supply a non-null owner.
   */
  ActionOwner getOwner();

  /**
   * Returns a mnemonic (string constant) for this kind of spawn.
   */
  String getMnemonic();
}
