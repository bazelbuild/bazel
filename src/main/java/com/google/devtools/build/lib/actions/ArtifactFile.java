// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * An ArtifactFile represents a file used by the build system during execution of Actions.
 * An ArtifactFile may be a source file or a derived (output) file.
 *
 * During the creation of {@link com.google.devtools.build.lib.analysis.ConfiguredTarget}s and
 * {@link com.google.devtools.build.lib.analysis.ConfiguredAspect}s, generally one
 * is only interested in {@link Artifact}s. Most Actions are only interested in Artifacts as well.
 * During action execution, however, some Artifacts (notably
 * TreeArtifacts) may correspond to more than one ArtifactFile. This is for the benefit
 * of some Actions which only know their inputs or outputs at execution time, via the
 * TreeArtifact mechanism.
 * <ul><li>
 *   Generally, most Artifacts represent a file on the filesystem, and correspond to exactly one
 *   ArtifactFile. For ease of use, Artifact itself implements ArtifactFile, and is modeled
 *   as an Artifact containing itself.
 * </li><li>
 *   Some Artifacts are so-called TreeArtifacts, for which the method
 *   {@link Artifact#isTreeArtifact()} returns true. These artifacts contain a (possibly empty)
 *   tree of ArtifactFiles, which are unknown until Action execution time.
 * </li><li>
 *   Some 'special artifacts' do not meaningfully represent a file or tree of files.
 * </li></ul>
 */
public interface ArtifactFile extends ActionInput {
  /**
   * Returns the exec path of this ArtifactFile. The exec path is a relative path
   * that is suitable for accessing this artifact relative to the execution
   * directory for this build.
   */
  PathFragment getExecPath();

  /**
   * Returns the path of this ArtifactFile relative to its containing Artifact.
   * See {@link #getParent()}.
   * */
  PathFragment getParentRelativePath();

  /** Returns the location of this ArtifactFile on the filesystem. */
  Path getPath();

  /**
   * Returns the relative path to this ArtifactFile relative to its root. Useful
   * when deriving output filenames from input files, etc.
   */
  PathFragment getRootRelativePath();

  /**
   * Returns the root beneath which this ArtifactFile resides, if any. This may be one of the
   * package-path entries (for files belonging to source {@link Artifact}s), or one of the bin
   * genfiles or includes dirs (for files belonging to derived {@link Artifact}s). It will always be
   * an ancestor of {@link #getPath()}.
   */
  Root getRoot();

  /**
   * Returns the Artifact containing this File. For Artifacts which are files, returns
   * {@code this}. Otherwise, returns a Artifact whose path and root relative path
   * are an ancestor of this ArtifactFile's path, and for which {@link Artifact#isTreeArtifact()}
   * returns true.
   * <p/>
   * For ArtifactFiles which are special artifacts (including TreeArtifacts),
   * this method throws UnsupportedOperationException, because the ArtifactFile abstraction
   * fails for those cases.
   */
  Artifact getParent() throws UnsupportedOperationException;

  /**
   * Returns a pretty string representation of the path denoted by this ArtifactFile, suitable for
   * use in user error messages. ArtifactFiles beneath a root will be printed relative to that root;
   * other ArtifactFiles will be printed as an absolute path.
   *
   * <p>(The toString method is intended for developer messages since its more informative.)
   */
  String prettyPrint();

  /**
   * Two ArtifactFiles are equal if their parent Artifacts and parent relative paths
   * are equal. If this ArtifactFile is itself an Artifact, see {@link Artifact#equals(Object)}.
   * Note that within a build, two ArtifactFiles produced by the build system will be equal
   * if and only if their full paths are equal.
   */
  @Override
  boolean equals(Object o);

  /**
   * An ArtifactFile's hash code should be equal to
   * {@code {@link #getParent().hashCode()} * 257 +
   * {@link #getParentRelativePath().hashCode()}}. If this ArtifactFile is itself an Artifact,
   * see {@link Artifact#hashCode()}.
   * <p/>
   * 257 is chosen because it is a Fermat prime, and has minimal 'bit overlap' with the Mersenne
   * prime of 31 used in {@link Path#hashCode()} and {@link String#hashCode()}.
   */
  @Override
  int hashCode();
}
