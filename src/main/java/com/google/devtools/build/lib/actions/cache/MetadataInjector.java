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
package com.google.devtools.build.lib.actions.cache;

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;

/** Supports metadata injection of action outputs into skyframe. */
public interface MetadataInjector {

  /**
   * Injects metadata of a file that is stored remotely.
   *
   * @param output a regular output file.
   * @param digest the digest of the file.
   * @param size the size of the file in bytes.
   * @param locationIndex is only used in Blaze.
   * @param actionId the id of the action that produced this file.
   */
  void injectRemoteFile(
      Artifact output, byte[] digest, long size, int locationIndex, String actionId);

  /**
   * Inject the metadata of a tree artifact whose contents are stored remotely.
   *
   * @param output an output directory.
   * @param children the metadata of the files stored in the directory. The paths must be relative
   *     to the path of {@code output}.
   */
  void injectRemoteDirectory(
      Artifact.SpecialArtifact output, Map<PathFragment, RemoteFileArtifactValue> children);

  /**
   * Marks an {@link Artifact} as intentionally omitted.
   *
   * <p>This is used as an optimization to not download "orphaned artifacts" (=artifacts that no
   * action depends on) from a remote system.
   */
  void markOmitted(ActionInput output);

  /**
   * Registers the given output as contents of a TreeArtifact, without injecting its digest. Prefer
   * {@link #injectDigest} when the digest is available.
   */
  void addExpandedTreeOutput(TreeFileArtifact output);

  /**
   * Injects provided digest into the metadata handler, simultaneously caching lstat() data as well.
   */
  void injectDigest(ActionInput output, FileStatus statNoFollow, byte[] digest);
}
