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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.starlarkbuildapi.FilesetEntryApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;

/**
 * FilesetEntry is a value object used to represent a "FilesetEntry" inside a "Fileset" BUILD rule.
 */
@Immutable
@ThreadSafe
public final class FilesetEntry implements StarlarkValue, FilesetEntryApi {

  private static final SymlinkBehavior DEFAULT_SYMLINK_BEHAVIOR = SymlinkBehavior.COPY;
  public static final String DEFAULT_STRIP_PREFIX = ".";
  public static final String STRIP_PREFIX_WORKSPACE = "%workspace%";

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("FilesetEntry(srcdir = ");
    printer.repr(srcLabel.toString());
    printer.append(", files = ");
    printer.repr(files == null ? ImmutableList.of() : Lists.transform(files, Label::toString));
    printer.append(", excludes = ");
    printer.repr(excludes == null ? ImmutableList.of() : excludes.asList());
    printer.append(", destdir = ");
    printer.repr(destDir.getPathString());
    printer.append(", strip_prefix = ");
    printer.repr(stripPrefix);
    printer.append(", symlinks = ");
    printer.repr(symlinkBehavior.toString());
    printer.append(")");
  }

  /** SymlinkBehavior decides what to do when a source file of a FilesetEntry is a symlink. */
  @Immutable
  @ThreadSafe
  public enum SymlinkBehavior {
    /** Just copies the symlink as-is. May result in dangling links. */
    COPY,
    /** Follow the link and make the destination point to the absolute path of the final target. */
    DEREFERENCE;

    public static SymlinkBehavior parse(String value) throws IllegalArgumentException {
      return valueOf(value.toUpperCase(Locale.ENGLISH));
    }

    @Override
    public String toString() {
      return super.toString().toLowerCase();
    }
  }

  private final Label srcLabel;
  @Nullable private final ImmutableList<Label> files;
  @Nullable private final ImmutableSet<String> excludes;
  private final PathFragment destDir;
  private final SymlinkBehavior symlinkBehavior;
  private final String stripPrefix;

  /**
   * Constructs a FilesetEntry with the given values.
   *
   * @param srcLabel the label of the source directory. Must be non-null.
   * @param files The explicit files to include. May be null.
   * @param excludes The files to exclude. Man be null. May only be non-null if files is null.
   * @param destDir The target-relative output directory.
   * @param symlinkBehavior how to treat symlinks on the input. See
   *        {@link FilesetEntry.SymlinkBehavior}.
   * @param stripPrefix the prefix to strip from the package-relative path. If ".", keep only the
   *        basename.
   */
  public FilesetEntry(
      Label srcLabel,
      @Nullable List<Label> files,
      @Nullable Collection<String> excludes,
      @Nullable String destDir,
      @Nullable SymlinkBehavior symlinkBehavior,
      @Nullable String stripPrefix) {
    this.srcLabel = Preconditions.checkNotNull(srcLabel);
    this.files = files == null ? null : ImmutableList.copyOf(files);
    this.excludes = (excludes == null || excludes.isEmpty()) ? null : ImmutableSet.copyOf(excludes);
    this.destDir = PathFragment.create((destDir == null) ? "" : destDir);
    this.symlinkBehavior = symlinkBehavior == null ? DEFAULT_SYMLINK_BEHAVIOR : symlinkBehavior;
    this.stripPrefix = stripPrefix == null ? DEFAULT_STRIP_PREFIX : stripPrefix;
  }

  /**
   * @return the source label.
   */
  public Label getSrcLabel() {
    return srcLabel;
  }

  /**
   * @return the destDir. Non null.
   */
  public PathFragment getDestDir() {
    return destDir;
  }

  /**
   * @return how symlinks should be handled.
   */
  public SymlinkBehavior getSymlinkBehavior() {
    return symlinkBehavior;
  }

  /**
   * @return an immutable set of excludes. Null if none specified.
   */
  @Nullable
  public ImmutableSet<String> getExcludes() {
    return excludes;
  }

  /**
   * @return an immutable list of file labels. Null if none specified.
   */
  @Nullable
  public ImmutableList<Label> getFiles() {
    return files;
  }

  /**
   * @return true if this Fileset should get files from the source directory.
   */
  public boolean isSourceFileset() {
    return "BUILD".equals(srcLabel.getName());
  }

  /**
   * @return all prerequisite labels in the FilesetEntry.
   */
  public Collection<Label> getLabels() {
    Set<Label> labels = new LinkedHashSet<>();
    if (files != null) {
      labels.addAll(files);
    } else {
      labels.add(srcLabel);
    }
    return labels;
  }

  /**
   * @return the prefix that should be stripped from package-relative path names.
   */
  public String getStripPrefix() {
    return stripPrefix;
  }

  /**
   * @return null if the entry is valid, and a human-readable error message otherwise.
   */
  @Nullable
  public String validate() {
    if (excludes != null && files != null) {
      return "Cannot specify both 'files' and 'excludes' in a FilesetEntry";
    } else if (files != null && !isSourceFileset()) {
      return "Cannot specify files with Fileset label '" + srcLabel + "'";
    } else if (destDir.isAbsolute()) {
      return "Cannot specify absolute destdir '" + destDir + "'";
    } else if (!stripPrefix.equals(DEFAULT_STRIP_PREFIX) && files == null) {
      return "If the strip prefix is not \"" + DEFAULT_STRIP_PREFIX + "\", files must be specified";
    } else if (stripPrefix.startsWith("/")) {
      return "Cannot specify absolute strip prefix; perhaps you need to use \""
          + STRIP_PREFIX_WORKSPACE + "\"";
    } else if (PathFragment.create(stripPrefix).containsUplevelReferences()) {
      return "Strip prefix must not contain uplevel references";
    } else if (stripPrefix.startsWith("%") && !stripPrefix.startsWith(STRIP_PREFIX_WORKSPACE)) {
      return "If the strip_prefix starts with \"%\" then it must start with \""
          + STRIP_PREFIX_WORKSPACE + "\"";
    } else {
      return null;
    }
  }

  @Override
  public String toString() {
    return String.format(
        "FilesetEntry(srcdir=%s, destdir=%s, strip_prefix=%s, symlinks=%s, "
            + "%d file(s) and %d excluded)",
        srcLabel,
        destDir,
        stripPrefix,
        symlinkBehavior,
        files != null ? files.size() : 0,
        excludes != null ? excludes.size() : 0);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(srcLabel, files, excludes, destDir, symlinkBehavior, stripPrefix);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }

    if (!(other instanceof FilesetEntry)) {
      return false;
    }

    FilesetEntry that = (FilesetEntry) other;
    return Objects.equal(srcLabel, that.srcLabel)
        && Objects.equal(files, that.files)
        && Objects.equal(excludes, that.excludes)
        && Objects.equal(destDir, that.destDir)
        && Objects.equal(symlinkBehavior, that.symlinkBehavior)
        && Objects.equal(stripPrefix, that.stripPrefix);
  }


}
