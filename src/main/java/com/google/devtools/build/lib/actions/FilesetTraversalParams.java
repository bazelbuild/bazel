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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.Set;

/**
 * Parameters of a filesystem traversal requested by a Fileset rule.
 *
 * <p>This object stores the details of the traversal request, e.g. whether it's a direct or nested
 * traversal (see {@link #getDirectTraversal()} and {@link #getNestedTraversal()}) or who the owner
 * of the traversal is.
 */
public interface FilesetTraversalParams {

  /** Desired behavior if the traversal hits a directory with a BUILD file, i.e. a subpackage. */
  public enum PackageBoundaryMode {
    /** The traversal should recurse into the directory, optionally reporting a warning. */
    CROSS,

    // TODO(bazel-team): deprecate CROSS and REPORT_ERROR in favor of DONT_CROSS. Clean up the depot
    // and lock down the semantics of FilesetEntry.srcdir to only accept other Filesets or BUILD
    // files of a package, in which case also require an explicit list of files.
    /** The traversal should not recurse into the directory but silently skip it. */
    DONT_CROSS,

    /** The traversal should not recurse into the directory and report an error. */
    REPORT_ERROR;

    public static PackageBoundaryMode forStrictFilesetFlag(boolean flagEnabled) {
      return flagEnabled ? REPORT_ERROR : CROSS;
    }

    public void fingerprint(Fingerprint fp) {
      fp.addInt(ordinal());
    }
  }

  /**
   * Abstraction of the root directory of a {@link DirectTraversal}.
   *
   * <ul>
   * <li>The root of package traversals is the package directory, i.e. the parent of the BUILD file.
   * <li>The root of "recursive" directory traversals is the directory's path.
   * <li>The root of "file" traversals is the path of the file (or directory, or symlink) itself.
   * </ul>
   *
   * <p>For the meaning of "recursive" and "file" traversals see {@link DirectTraversal}.
   */
  interface DirectTraversalRoot {

    /**
     * Returns the root part of the full path.
     *
     * <p>This is typically the workspace root or some output tree's root (e.g. genfiles, binfiles).
     */
    Path getRootPart();

    /**
     * Returns the {@link #getRootPart() root}-relative part of the path.
     *
     * <p>This is typically the source directory under the workspace or the output file under an
     * output directory.
     */
    PathFragment getRelativePart();

    /** Returns a {@link RootedPath} composed of the root and relative parts. */
    RootedPath asRootedPath();
  }

  /**
   * Describes a request for a direct filesystem traversal.
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
  interface DirectTraversal {

    /** Returns the root of the traversal; see {@link DirectTraversalRoot}. */
    DirectTraversalRoot getRoot();

    /**
     * Returns true if this traversal refers to a whole package.
     *
     * <p>In that case the root (see {@link #getRoot()}) refers to the path of the package.
     *
     * <p>Package traversals are always recursive (see {@link #isRecursive()}) and are never
     * generated (see {@link #isGenerated()}).
     */
    boolean isPackage();

    /**
     * Returns true if this is a "recursive traversal", i.e. created from FilesetEntry.srcdir.
     *
     * <p>This type of traversal is created when the FilesetEntry doesn't define a "files" list.
     * When it does, the traversal is referred to as a "file traversal". When it doesn't, but the
     * srcdir points to another Fileset, it is called a "nested" traversal.
     *
     * <p>Recursive traversals got their name from recursively traversing a directory structure.
     * These are usually whole-package traversals, i.e. when FilesetEntry.srcdir refers to a BUILD
     * file (see {@link #isPackage()}), but sometimes the srcdir references a input or output
     * directory (the latter being generated by a local genrule) or a symlink (which must point to a
     * directory; enforced during action execution).
     *
     * <p>The files in the results of a recursive traversal are all under the {@link #getRoot()
     * root}. The root's path is stripped from the results.
     *
     * <p>N.B.: "file traversals" can also be recursive if the entry in FilesetEntry.files, for
     * which the traversal parameters were created, turned out to be a directory. The difference
     * lies in how the output paths are computed (with recursive traversals, the directory's name
     * is stripped; with file traversals it is not, modulo usage of strip_prefix and the excludes
     * attributes), and how directory symlinks are handled (in "recursive traversals" they are
     * expanded just like normal directories, subsequent directory symlinks under them are *not*
     * expanded though; they are not expanded at all in "file traversals").
     */
    boolean isRecursive();

    /** Returns true if the root points to a generated file, symlink or directory. */
    boolean isGenerated();

    /** Returns true if input symlinks should be dereferenced; false if copied. */
    boolean isFollowingSymlinks();

    /** Returns the desired behavior when the traversal hits a subpackage. */
    PackageBoundaryMode getPackageBoundaryMode();
  }

  /** Label of the Fileset rule that owns this traversal. */
  Label getOwnerLabelForErrorMessages();

  /** Returns the directory under the output path where the files will be mapped. May be empty. */
  PathFragment getDestPath();

  /** Returns a list of file basenames to be excluded from the output. May be empty. */
  Set<String> getExcludedFiles();

  /**
   * Returns the parameters of the direct traversal request, if any.
   *
   * <p>A direct traversal is anything that's not a nested traversal, e.g. traversal of a package or
   * directory (when FilesetEntry.srcdir is specified) or traversal of a single file (when
   * FilesetEntry.files is specified). See {@link DirectTraversal} for more detail.
   *
   * <p>The value is present if and only if {@link #getNestedTraversal} is empty.
   */
  Optional<DirectTraversal> getDirectTraversal();

  /**
   * Returns the parameters of the nested traversal request, if any.
   *
   * <p>A nested traversal is the traversal of another Fileset referenced by FilesetEntry.srcdir.
   *
   * <p>The value is non-empty when {@link #getDirectTraversal} is absent AND the nested Fileset has
   * non-empty FilesetEntries.
   */
  ImmutableList<FilesetTraversalParams> getNestedTraversal();

  /** Adds the fingerprint of this traversal object. */
  void fingerprint(Fingerprint fp);
}
