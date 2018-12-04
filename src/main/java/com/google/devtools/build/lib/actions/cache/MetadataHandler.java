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

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.vfs.FileStatus;
import java.io.IOException;

/**
 * Retrieves {@link FileArtifactValue} of {@link Artifact}s, and inserts virtual metadata as well.
 *
 * <p>Some methods on this interface may only be called after a call to {@link
 * #discardOutputMetadata}. Calling them before such a call results in an {@link
 * IllegalStateException}.
 *
 * <p>Note that implementations of this interface call chmod on output files if {@link
 * #discardOutputMetadata} has been called.
 */
public interface MetadataHandler extends MetadataProvider, MetadataInjector {
  @Override
  FileArtifactValue getMetadata(ActionInput actionInput) throws IOException;

  /** Sets digest for virtual artifacts (e.g. middlemen). {@code md5Digest} must not be null. */
  void setDigestForVirtualArtifact(Artifact artifact, Md5Digest md5Digest);

  /**
   * Registers the given output as contents of a TreeArtifact, without injecting its digest. Prefer
   * {@link #injectDigest} when the digest is available.
   *
   * <p>Must only be called after a call to {@link #discardOutputMetadata}.
   */
  void addExpandedTreeOutput(TreeFileArtifact output);

  /** Retrieves the artifacts inside the TreeArtifact, without injecting its digest. */
  Iterable<TreeFileArtifact> getExpandedOutputs(Artifact artifact);

  /**
   * Injects provided digest into the metadata handler, simultaneously caching lstat() data as well.
   *
   * <p>Must only be called after a call to {@link #discardOutputMetadata}.
   */
  void injectDigest(ActionInput output, FileStatus statNoFollow, byte[] digest);

  /**
   * Marks an artifact as intentionally omitted. Acknowledges that this Artifact could have existed,
   * but was intentionally not saved, most likely as an optimization.
   *
   * <p>Must only be called after a call to {@link #discardOutputMetadata}.
   */
  void markOmitted(ActionInput output);

  /**
   * Returns true iff artifact was intentionally omitted (not saved).
   */
  // TODO(ulfjack): artifactOmitted always returns false unless we've just executed the action, and
  // made calls to markOmitted. We either need to document that or change it so it works reliably.
  boolean artifactOmitted(Artifact artifact);

  /**
   * Discards all known output artifact metadata, presumably because outputs will be modified. May
   * only be called before any metadata is injected using {@link #injectDigest} or {@link
   * #markOmitted};
   *
   * <p>Must be called at most once on any specific instance.
   */
  void discardOutputMetadata();
}
