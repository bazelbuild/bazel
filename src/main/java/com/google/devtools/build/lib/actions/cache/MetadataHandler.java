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
import com.google.devtools.build.lib.actions.ArtifactFile;
import com.google.devtools.build.lib.actions.MiddlemanAction;
import com.google.devtools.build.lib.vfs.FileStatus;

import java.io.IOException;

/** Retrieves {@link Metadata} of {@link Artifact}s, and inserts virtual metadata as well. */
public interface MetadataHandler {
  /**
   * Returns metadata for the given artifact or null if it does not exist or is intentionally
   * omitted.
   *
   * <p>This should always be used for the inputs to {@link MiddlemanAction}s instead of
   * {@link #getMetadata(Artifact)} since we may allow non-existent inputs to middlemen.</p>
   *
   * @param artifact artifact
   *
   * @return metadata instance or null if metadata cannot be obtained.
   */
  Metadata getMetadataMaybe(Artifact artifact);

  /**
   * Returns metadata for the given artifact or throws an exception if the
   * metadata could not be obtained.
   *
   * @return metadata instance
   *
   * @throws IOException if metadata could not be obtained.
   */
  Metadata getMetadata(Artifact artifact) throws IOException;

  /** Sets digest for virtual artifacts (e.g. middlemen). {@code digest} must not be null. */
  void setDigestForVirtualArtifact(Artifact artifact, Digest digest);

  /**
   * Registers the given output as contents of a TreeArtifact, without injecting its digest.
   * Prefer {@link #injectDigest} when the digest is available.
   * @throws IllegalStateException if the given output does not have a TreeArtifact parent.
   */
  void addExpandedTreeOutput(ArtifactFile output) throws IllegalStateException;

  /**
   * Injects provided digest into the metadata handler, simultaneously caching lstat() data as well.
   */
  void injectDigest(ActionInput output, FileStatus statNoFollow, byte[] digest);

  /**
   * Marks an artifact as intentionally omitted. Acknowledges that this Artifact could have
   * existed, but was intentionally not saved, most likely as an optimization.
   */
  void markOmitted(ActionInput output);

  /**
   * Returns true iff artifact exists.
   *
   * <p>It is important to note that implementations may cache non-existence as a side effect
   * of this method. If there is a possibility an artifact was intentionally omitted then
   * {@link #artifactOmitted(Artifact)} should be checked first to avoid the side effect.</p>
   */
  boolean artifactExists(Artifact artifact);

  /** Returns true iff artifact is a regular file. */
  boolean isRegularFile(Artifact artifact);

  /** Returns true iff artifact was intentionally omitted (not saved). */
  boolean artifactOmitted(Artifact artifact);

  /**
   * @return Whether the ArtifactFile's data was injected.
   * @throws IOException if implementation tried to stat the ArtifactFile which threw an exception.
   *         Technically, this means that the artifact could not have been injected, but by throwing
   *         here we save the caller trying to stat this file on their own and throwing the same
   *         exception. Implementations are not guaranteed to throw in this case if they are able to
   *         determine that the artifact is not injected without statting it.
   */
  boolean isInjected(ArtifactFile file) throws IOException;

  /**
   * Discards all known output artifact metadata, presumably because outputs will be modified.
   * May only be called before any metadata is injected using {@link #injectDigest} or
   * {@link #markOmitted};
   */
  void discardOutputMetadata();

}
