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
package com.google.devtools.build.lib.actions.cache;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.vfs.FileStatus;
import java.io.IOException;
import javax.annotation.Nullable;

/** Handles metadata of inputs and outputs during the execution phase. */
public interface MetadataHandler extends MetadataProvider, MetadataInjector {

  /**
   * {@inheritDoc}
   *
   * <p>Freshly created output files (i.e. from an action that just executed) that require a stat to
   * obtain the metadata will first be set read-only and executable during this call. This ensures
   * that the returned metadata has an appropriate ctime, which is affected by chmod. Note that this
   * does not apply to outputs injected via {@link #injectFile} or {@link #injectDirectory} since a
   * stat is not required for them.
   */
  @Override
  @Nullable
  FileArtifactValue getMetadata(ActionInput input) throws IOException;

  /** Sets digest for virtual artifacts (e.g. middlemen). {@code digest} must not be null. */
  void setDigestForVirtualArtifact(Artifact artifact, byte[] digest);

  /**
   * Constructs a {@link FileArtifactValue} for the given output whose digest is known.
   *
   * <p>This call does not inject the returned metadata. It should be injected with a followup call
   * to {@link #injectFile} or {@link #injectDirectory} as appropriate.
   *
   * <p>chmod will not be called on the output.
   */
  FileArtifactValue constructMetadataForDigest(
      Artifact output, FileStatus statNoFollow, byte[] injectedDigest) throws IOException;

  /**
   * Retrieves the children of a tree artifact, returning an empty set if there is no data
   * available.
   */
  ImmutableSet<TreeFileArtifact> getTreeArtifactChildren(SpecialArtifact treeArtifact);

  /**
   * Marks an {@link Artifact} as intentionally omitted.
   *
   * <p>This is used as an optimization to not download <em>orphaned</em> artifacts (artifacts that
   * no action depends on) from a remote system.
   */
  void markOmitted(Artifact output);

  /** Returns {@code true} if {@link #markOmitted} was called on the artifact. */
  boolean artifactOmitted(Artifact artifact);

  /**
   * Discards any cached metadata for the given outputs.
   *
   * <p>May be called if an action can make multiple attempts that are expected to create the same
   * set of output files.
   */
  void resetOutputs(Iterable<Artifact> outputs);
}
