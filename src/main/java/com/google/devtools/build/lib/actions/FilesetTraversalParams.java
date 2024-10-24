// Copyright 2015 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.actions;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.Instantiator;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Parameters of a filesystem traversal requested by a Fileset rule.
 *
 * <p>This object stores the details of the traversal request, e.g. whether it's a direct or nested
 * traversal (see {@link #getDirectTraversal()} and {@link #getNestedArtifact()}) or who the owner
 * of the traversal is.
 */
public interface FilesetTraversalParams {

  /**
   * The root directory of a {@link DirectTraversal}.
   *
   * <ul>
   *   <li>The root of package traversals is the package directory, i.e. the parent of the BUILD
   *       file.
   *   <li>The root of "recursive" directory traversals is the directory's path.
   *   <li>The root of "file" traversals is the path of the file (or directory, or symlink) itself.
   * </ul>
   *
   * <p>For the meaning of "recursive" and "file" traversals see {@link DirectTraversal}.
   */
  @AutoValue
  abstract class DirectTraversalRoot {

    /**
     * Returns the output Artifact corresponding to this traversal, if present. Only present when
     * traversing a generated output.
     */
    @Nullable
    public abstract Artifact getOutputArtifact();

    /**
     * Returns the root part of the full path.
     *
     * <p>This is typically the workspace root or some output tree's root (e.g. genfiles, binfiles).
     */
    public abstract Root getRootPart();

    /**
     * Returns the {@link #getRootPart() root}-relative part of the path.
     *
     * <p>This is typically the source directory under the workspace or the output file under an
     * output directory.
     */
    public abstract PathFragment getRelativePart();

    /** Returns a {@link Path} composed of the root and relative parts. */
    public final Path asPath() {
      return getRootPart().getRelative(getRelativePart());
    }

    /** Returns a {@link RootedPath} composed of the root and relative parts. */
    public final RootedPath asRootedPath() {
      return RootedPath.toRootedPath(getRootPart(), getRelativePart());
    }

    @Override
    public final boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (o instanceof DirectTraversalRoot that) {
        return Objects.equals(this.getOutputArtifact(), that.getOutputArtifact())
            && this.getRootPart().equals(that.getRootPart())
            && this.getRelativePart().equals(that.getRelativePart());
      }
      return false;
    }

    @Memoized
    @Override
    public abstract int hashCode();

    public static DirectTraversalRoot forFileOrDirectory(Artifact fileOrDirectory) {
      return create(
          fileOrDirectory.isSourceArtifact() ? null : fileOrDirectory,
          fileOrDirectory.getRoot().getRoot(),
          fileOrDirectory.getRootRelativePath());
    }

    public static DirectTraversalRoot forRootedPath(RootedPath rootedPath) {
      return forRootAndPath(rootedPath.getRoot(), rootedPath.getRootRelativePath());
    }

    public static DirectTraversalRoot forRootAndPath(Root rootPart, PathFragment relativePart) {
      return create(/*outputArtifact=*/ null, rootPart, relativePart);
    }

    @Instantiator
    @VisibleForSerialization
    static DirectTraversalRoot create(
        @Nullable Artifact outputArtifact, Root rootPart, PathFragment relativePart) {
      return new AutoValue_FilesetTraversalParams_DirectTraversalRoot(
          outputArtifact, rootPart, relativePart);
    }
  }

  /**
   * A request for a direct filesystem traversal.
   *
   * <p>"Direct" means this corresponds to an actual filesystem traversal as opposed to traversing
   * another Fileset rule, which is called a "nested" traversal.
   *
   * <p>Direct traversals can further be divided into two categories, "file" traversals and
   * "recursive" traversals.
   *
   * <p>File traversal requests are created when the FilesetEntry.files attribute is defined; one
   * file traversal request is created for each entry.
   *
   * <p>Recursive traversal requests are created when the FilesetEntry.files attribute is
   * unspecified; one recursive traversal request is created for the FilesetEntry.srcdir.
   *
   * <p>See {@link DirectTraversal#getRoot()} for more details.
   */
  @AutoValue
  abstract class DirectTraversal {

    /** Returns the root of the traversal; see {@link DirectTraversalRoot}. */
    public abstract DirectTraversalRoot getRoot();

    /** Returns true if the root points to a generated file, symlink or directory. */
    public abstract boolean isGenerated();

    /** Returns whether Filesets treat outputs in a strict manner, assuming regular files. */
    public abstract boolean isStrictFilesetOutput();

    /** Returns whether directories are permitted and can be traversed. */
    public abstract boolean permitDirectories();

    @Memoized
    @Override
    public abstract int hashCode();

    @Memoized
    byte[] getFingerprint() {
      Fingerprint fp = new Fingerprint();
      fp.addPath(getRoot().asPath());
      fp.addBoolean(isGenerated());
      fp.addBoolean(isStrictFilesetOutput());
      fp.addBoolean(permitDirectories());
      return fp.digestAndReset();
    }

    static DirectTraversal getDirectTraversal(
        DirectTraversalRoot root,
        boolean isStrictFilesetOutput,
        boolean permitSourceDirectories,
        boolean isGenerated) {
      return new AutoValue_FilesetTraversalParams_DirectTraversal(
          root, isGenerated, isStrictFilesetOutput, permitSourceDirectories);
    }
  }

  /** Label of the Fileset rule that owns this traversal. */
  Label getOwnerLabelForErrorMessages();

  /** Returns the directory under the output path where the files will be mapped. May be empty. */
  PathFragment getDestPath();

  /** Returns a list of file basenames to be excluded from the output. May be empty. */
  ImmutableSortedSet<String> getExcludedFiles();

  /**
   * Returns the parameters of the direct traversal request, if any.
   *
   * <p>A direct traversal is anything that's not a nested traversal, e.g. traversal of a package or
   * directory (when FilesetEntry.srcdir is specified) or traversal of a single file (when
   * FilesetEntry.files is specified). See {@link DirectTraversal} for more detail.
   *
   * <p>The returned value is non-null if and only if {@link #getNestedArtifact} is null.
   */
  @Nullable
  DirectTraversal getDirectTraversal();

  /**
   * Returns the Fileset Artifact of the nested traversal request, if any.
   *
   * <p>A nested traversal is the traversal of another Fileset referenced by FilesetEntry.srcdir.
   *
   * <p>The value is non-null when {@link #getDirectTraversal} is absent.
   */
  @Nullable
  Artifact getNestedArtifact();

  /** Adds the fingerprint of this traversal object. */
  void fingerprint(Fingerprint fp);
}
