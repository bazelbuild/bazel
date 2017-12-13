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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.Comparator.comparing;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.EvalUtils.ComparisonException;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * An Artifact represents a file used by the build system, whether it's a source
 * file or a derived (output) file. Not all Artifacts have a corresponding
 * FileTarget object in the <code>build.lib.packages</code> API: for example,
 * low-level intermediaries internal to a given rule, such as a Java class files
 * or C++ object files. However all FileTargets have a corresponding Artifact.
 *
 * <p>In any given call to SkyframeExecutor#buildArtifacts(), no two Artifacts in the
 * action graph may refer to the same path.
 *
 * <p>Artifacts generally fall into two classifications, source and derived, but
 * there exist a few other cases that are fuzzy and difficult to classify. The
 * following cases exist:
 * <ul>
 * <li>Well-formed source Artifacts will have null generating Actions and a root
 * that is orthogonal to execRoot. (With the root coming from the package path.)
 * <li>Well-formed derived Artifacts will have non-null generating Actions, and
 * a root that is below execRoot.
 * <li>Symlinked include source Artifacts under the output/include tree will
 * appear to be derived artifacts with null generating Actions.
 * <li>Some derived Artifacts, mostly in the genfiles tree and mostly discovered
 * during include validation, will also have null generating Actions.
 * </ul>
 *
 * In the usual case, an Artifact represents a single file. However, an Artifact may
 * also represent the following:
 * <ul>
 * <li>A TreeArtifact, which is a directory containing a tree of unknown {@link Artifact}s.
 * In the future, Actions will be able to examine these files as inputs and declare them as outputs
 * at execution time, but this is not yet implemented. This is used for Actions where
 * the inputs and/or outputs might not be discoverable except during Action execution.
 * <li>A directory of unknown contents, but not a TreeArtifact.
 * This is a legacy facility and should not be used by any new rule implementations.
 * In particular, the file system cache integrity checks fail for directories.
 * <li>An 'aggregating middleman' special Artifact, which may be expanded using a
 * {@link ArtifactExpander} at Action execution time. This is used by a handful of rules to save
 * memory.
 * <li>A 'constant metadata' special Artifact. These represent real files, changes to which are
 * ignored by the build system. They are useful for files which change frequently but do not affect
 * the result of a build, such as timestamp files.
 * <li>A 'Fileset' special Artifact. This is a legacy type of Artifact and should not be used
 * by new rule implementations.
 * </ul>
 * <p>This class is "theoretically" final; it should not be subclassed except by
 * {@link SpecialArtifact}.
 */
@Immutable
@SkylarkModule(name = "File",
    category = SkylarkModuleCategory.BUILTIN,
    doc = "<p>This type represents a file or directory used by the build system. It can be "
        + "either a source file or a derived file produced by a rule.</p>"
        + "<p>The File constructor is private, so you cannot call it directly to create new "
        + "Files. If you have a Skylark rule that needs to create a new File, you have two options:"
        + "<ul>"
        + "<li>use <a href='actions.html#declare_file'>ctx.actions.declare_file</a> "
        + "or <a href='actions.html#declare_directory'>ctx.actions.declare_directory</a> to "
        + "declare a new file in the rule implementation.</li>"
        + "<li>add the label to the attrs (if it's an input) or the outputs (if it's an output)."
        + " Then you can access the File through the rule's "
        + "<a href='ctx.html#outputs'>ctx.outputs</a>.")
public class Artifact
    implements FileType.HasFilename, ActionInput, SkylarkValue, Comparable<Object> {

  /** Compares artifact according to their exec paths. Sorts null values first. */
  @SuppressWarnings("ReferenceEquality")  // "a == b" is an optimization
  public static final Comparator<Artifact> EXEC_PATH_COMPARATOR =
      (a, b) -> {
        if (a == b) {
          return 0;
        } else if (a == null) {
          return -1;
        } else if (b == null) {
          return -1;
        } else {
          return a.execPath.compareTo(b.execPath);
        }
      };

  /** Compares artifacts according to their root relative paths. */
  public static final Comparator<Artifact> ROOT_RELATIVE_PATH_COMPARATOR =
      comparing(Artifact::getRootRelativePath);

  @Override
  public int compareTo(Object o) {
    if (o instanceof Artifact) {
      return EXEC_PATH_COMPARATOR.compare(this, (Artifact) o);
    }
    throw new ComparisonException("Cannot compare artifact with " + EvalUtils.getDataTypeName(o));
  }


  /** An object that can expand middleman artifacts. */
  public interface ArtifactExpander {

    /**
     * Expands the given artifact, and populates "output" with the result.
     *
     * <p>{@code artifact.isMiddlemanArtifact() || artifact.isTreeArtifact()} must be true.
     * Only aggregating middlemen and tree artifacts are expanded.
     */
    void expand(Artifact artifact, Collection<? super Artifact> output);
  }

  public static final ImmutableList<Artifact> NO_ARTIFACTS = ImmutableList.of();

  /** A Predicate that evaluates to true if the Artifact is not a middleman artifact. */
  public static final Predicate<Artifact> MIDDLEMAN_FILTER = input -> !input.isMiddlemanArtifact();

  private final int hashCode;
  private final Path path;
  private final Root root;
  private final PathFragment execPath;
  private final PathFragment rootRelativePath;
  private final ArtifactOwner owner;

  /**
   * Constructs an artifact for the specified path, root and execPath. The root must be an ancestor
   * of path, and execPath must be a non-absolute tail of path. Outside of testing, this method
   * should only be called by ArtifactFactory. The ArtifactOwner may be null.
   *
   * <p>In a source Artifact, the path tail after the root will be identical to the execPath, but
   * the root will be orthogonal to execRoot.
   * <pre>
   *  [path] == [/root/][execPath]
   * </pre>
   *
   * <p>In a derived Artifact, the execPath will overlap with part of the root, which in turn will
   * be below the execRoot.
   * <pre>
   *  [path] == [/root][pathTail] == [/execRoot][execPath] == [/execRoot][rootPrefix][pathTail]
   * </pre>
   */
  @VisibleForTesting
  public Artifact(Path path, Root root, PathFragment execPath, ArtifactOwner owner) {
    if (root == null || !path.startsWith(root.getPath())) {
      throw new IllegalArgumentException(root + ": illegal root for " + path
          + " (execPath: " + execPath + ")");
    }
    if (execPath == null || execPath.isAbsolute() || !path.asFragment().endsWith(execPath)) {
      throw new IllegalArgumentException(execPath + ": illegal execPath for " + path
          + " (root: " + root + ")");
    }
    this.hashCode = path.hashCode();
    this.path = path;
    this.root = root;
    this.execPath = execPath;
    // These two lines establish the invariant that
    // execPath == rootRelativePath <=> execPath.equals(rootRelativePath)
    // This is important for isSourceArtifact.
    PathFragment rootRel = path.relativeTo(root.getPath());
    if (!execPath.endsWith(rootRel)) {
      throw new IllegalArgumentException(execPath + ": illegal execPath doesn't end with "
          + rootRel + " at " + path + " with root " + root);
    }
    this.rootRelativePath = rootRel.equals(execPath) ? execPath : rootRel;
    this.owner = Preconditions.checkNotNull(owner, path);
  }

  /**
   * Constructs an artifact for the specified path, root and execPath. The root must be an ancestor
   * of path, and execPath must be a non-absolute tail of path. Should only be called for testing.
   *
   * <p>In a source Artifact, the path tail after the root will be identical to the execPath, but
   * the root will be orthogonal to execRoot.
   * <pre>
   *  [path] == [/root/][execPath]
   * </pre>
   *
   * <p>In a derived Artifact, the execPath will overlap with part of the root, which in turn will
   * be below of the execRoot.
   * <pre>
   *  [path] == [/root][pathTail] == [/execRoot][execPath] == [/execRoot][rootPrefix][pathTail]
   * <pre>
   */
  @VisibleForTesting
  public Artifact(Path path, Root root, PathFragment execPath) {
    this(path, root, execPath, ArtifactOwner.NULL_OWNER);
  }

  /**
   * Constructs a source or derived Artifact for the specified path and specified root. The root
   * must be an ancestor of the path.
   */
  @VisibleForTesting  // Only exists for testing.
  public Artifact(Path path, Root root) {
    this(path, root, root.getExecPath().getRelative(path.relativeTo(root.getPath())),
        ArtifactOwner.NULL_OWNER);
  }

  /**
   * Constructs a source or derived Artifact for the specified root-relative path and root.
   */
  @VisibleForTesting  // Only exists for testing.
  public Artifact(PathFragment rootRelativePath, Root root) {
    this(root.getPath().getRelative(rootRelativePath), root,
        root.getExecPath().getRelative(rootRelativePath), ArtifactOwner.NULL_OWNER);
  }

  public final Path getPath() {
    return path;
  }

  public boolean hasParent() {
    return getParent() != null;
  }

  /**
   * Returns the parent Artifact containing this Artifact. Artifacts without parents shall
   * return null.
   */
  @Nullable public Artifact getParent() {
    return null;
  }

  /**
   * Returns the directory name of this artifact, similar to dirname(1).
   *
   * <p> The directory name is always a relative path to the execution directory.
   */
  @SkylarkCallable(name = "dirname", structField = true,
      doc = "The name of the directory containing this file.")
  public final String getDirname() {
    PathFragment parent = getExecPath().getParentDirectory();
    return (parent == null) ? "/" : parent.getSafePathString();
  }

  /**
   * Returns the base file name of this artifact, similar to basename(1).
   */
  @Override
  @SkylarkCallable(name = "basename", structField = true,
      doc = "The base name of this file. This is the name of the file inside the directory.")
  public final String getFilename() {
    return getExecPath().getBaseName();
  }

  @SkylarkCallable(name = "extension", structField = true, doc = "The file extension of this file.")
  public final String getExtension() {
    return getExecPath().getFileExtension();
  }

  /**
   * Returns the artifact owner. May be null.
   */
  @Nullable public final Label getOwner() {
    return owner.getLabel();
  }

  /**
   * Get the {@code LabelAndConfiguration} of the {@code ConfiguredTarget} that owns this artifact,
   * if it was set. Otherwise, this should be a dummy value -- either {@link
   * ArtifactOwner#NULL_OWNER} or a dummy owner set in tests. Such a dummy value should only occur
   * for source artifacts if created without specifying the owner, or for special derived artifacts,
   * such as target completion middleman artifacts, build info artifacts, and the like.
   */
  public final ArtifactOwner getArtifactOwner() {
    return owner;
  }

  @SkylarkCallable(name = "owner", structField = true, allowReturnNones = true,
    doc = "A label of a target that produces this File."
  )
  public Label getOwnerLabel() {
    return owner.getLabel();
  }

  /**
   * Returns the root beneath which this Artifact resides, if any. This may be one of the
   * package-path entries (for source Artifacts), or one of the bin, genfiles or includes dirs
   * (for derived Artifacts). It will always be an ancestor of getPath().
   */
  @SkylarkCallable(name = "root", structField = true,
      doc = "The root beneath which this file resides."
  )
  public final Root getRoot() {
    return root;
  }

  @Override
  public final PathFragment getExecPath() {
    return execPath;
  }

  /**
   * Returns the path of this Artifact relative to this containing Artifact. Since
   * ordinary Artifacts correspond to only one Artifact -- itself -- for ordinary Artifacts,
   * this just returns the empty path. For special Artifacts, throws
   * {@link UnsupportedOperationException}. See also {@link Artifact#getParentRelativePath()}.
   */
  public PathFragment getParentRelativePath() {
    return PathFragment.EMPTY_FRAGMENT;
  }

  /**
   * Returns true iff this is a source Artifact as determined by its path and root relationships.
   * Note that this will report all Artifacts in the output tree, including in the include symlink
   * tree, as non-source.
   */
  @SkylarkCallable(
    name = "is_source",
    structField = true,
    doc = "Returns true if this is a source file, i.e. it is not generated."
  )
  @SuppressWarnings("ReferenceEquality")  // == is an optimization
  public final boolean isSourceArtifact() {
    return execPath == rootRelativePath;
  }

  /**
   * Returns true iff this is a middleman Artifact as determined by its root.
   */
  public final boolean isMiddlemanArtifact() {
    return getRoot().isMiddlemanRoot();
  }

  /**
   * Returns true iff this is a TreeArtifact representing a directory tree containing Artifacts.
   */
  // TODO(rduan): Document this Skylark method once TreeArtifact is no longer experimental.
  @SkylarkCallable(name = "is_directory", structField = true, documented = false)
  public boolean isTreeArtifact() {
    return false;
  }

  /**
   * Returns whether the artifact represents a Fileset.
   */
  public boolean isFileset() {
    return false;
  }

  /**
   * Returns true iff metadata cache must return constant metadata for the
   * given artifact.
   */
  public boolean isConstantMetadata() {
    return false;
  }

  /**
   * Special artifact types.
   *
   * @see SpecialArtifact
   */
  @VisibleForTesting
  public enum SpecialArtifactType {
    FILESET,
    TREE,
    CONSTANT_METADATA,
  }

  /**
   * A special kind of artifact that either is a fileset or needs special metadata caching behavior.
   *
   * <p>We subclass {@link Artifact} instead of storing the special attributes inside in order
   * to save memory. The proportion of artifacts that are special is very small, and by not having
   * to keep around the attribute for the rest we save some memory.
   */
  @Immutable
  @VisibleForTesting
  public static final class SpecialArtifact extends Artifact {
    private final SpecialArtifactType type;

    @VisibleForTesting
    public SpecialArtifact(Path path, Root root, PathFragment execPath, ArtifactOwner owner,
        SpecialArtifactType type) {
      super(path, root, execPath, owner);
      this.type = type;
    }

    @Override
    public final boolean isFileset() {
      return type == SpecialArtifactType.FILESET;
    }

    @Override
    public boolean isConstantMetadata() {
      return type == SpecialArtifactType.CONSTANT_METADATA;
    }

    @Override
    public boolean isTreeArtifact() {
      return type == SpecialArtifactType.TREE;
    }

    @Override
    public boolean hasParent() {
      return false;
    }

    @Override
    @Nullable
    public Artifact getParent() {
      return null;
    }

    @Override
    @Nullable
    public PathFragment getParentRelativePath() {
      return null;
    }
  }

  /**
   * A special kind of artifact that represents a concrete file created at execution time under
   * its associated TreeArtifact.
   *
   * <p> TreeFileArtifacts should be only created during execution time inside some special actions
   * to support action inputs and outputs that are unpredictable at analysis time.
   * TreeFileArtifacts should not be created directly by any rules at analysis time.
   *
   * <p>We subclass {@link Artifact} instead of storing the extra fields directly inside in order
   * to save memory. The proportion of TreeFileArtifacts is very small, and by not having to keep
   * around the extra fields for the rest we save some memory.
   */
  @Immutable
  public static final class TreeFileArtifact extends Artifact {
    private final Artifact parentTreeArtifact;
    private final PathFragment parentRelativePath;

    /**
     * Constructs a TreeFileArtifact with the given parent-relative path under the given parent
     * TreeArtifact. The {@link ArtifactOwner} of the TreeFileArtifact is the {@link ArtifactOwner}
     * of the parent TreeArtifact.
     */
    TreeFileArtifact(Artifact parent, PathFragment parentRelativePath) {
      this(parent, parentRelativePath, parent.getArtifactOwner());
    }

    /**
     * Constructs a TreeFileArtifact with the given parent-relative path under the given parent
     * TreeArtifact, owned by the given {@code artifactOwner}.
     */
    TreeFileArtifact(Artifact parent, PathFragment parentRelativePath,
        ArtifactOwner artifactOwner) {
      super(
          parent.getPath().getRelative(parentRelativePath),
          parent.getRoot(),
          parent.getExecPath().getRelative(parentRelativePath),
          artifactOwner);
      Preconditions.checkState(
          parent.isTreeArtifact(),
          "The parent of TreeFileArtifact (parent-relative path: %s) is not a TreeArtifact: %s",
          parentRelativePath,
          parent);
      Preconditions.checkState(
          parentRelativePath.isNormalized() && !parentRelativePath.isAbsolute(),
          "%s is not a proper normalized relative path",
          parentRelativePath);
      this.parentTreeArtifact = parent;
      this.parentRelativePath = parentRelativePath;
    }

    @Override
    public Artifact getParent() {
      return parentTreeArtifact;
    }

    @Override
    public PathFragment getParentRelativePath() {
      return parentRelativePath;
    }
  }

  /**
   * Returns the relative path to this artifact relative to its root.  (Useful
   * when deriving output filenames from input files, etc.)
   */
  public final PathFragment getRootRelativePath() {
    return rootRelativePath;
  }

  /**
   * For targets in external repositories, this returns the path the artifact live at in the
   * runfiles tree. For local targets, it returns the rootRelativePath.
   */
  public final PathFragment getRunfilesPath() {
    PathFragment relativePath = rootRelativePath;
    if (relativePath.startsWith(Label.EXTERNAL_PATH_PREFIX)) {
      // Turn external/repo/foo into ../repo/foo.
      relativePath = relativePath.relativeTo(Label.EXTERNAL_PATH_PREFIX);
      relativePath = PathFragment.create("..").getRelative(relativePath);
    }
    return relativePath;
  }

  @SkylarkCallable(
      name = "short_path",
      structField = true,
      doc =
          "The path of this file relative to its root. This excludes the aforementioned "
              + "<i>root</i>, i.e. configuration-specific fragments of the path. This is also the "
              + "path under which the file is mapped if it's in the runfiles of a binary."
  )
  public final String getRunfilesPathString() {
    return getRunfilesPath().getPathString();
  }

  /**
   * Returns this.getExecPath().getPathString().
   */
  @Override
  @SkylarkCallable(
    name = "path",
    structField = true,
    doc =
        "The execution path of this file, relative to the workspace's execution directory. It "
            + "consists of two parts, an optional first part called the <i>root</i> (see also the "
            + "<a href=\"root.html\">root</a> module), and the second part which is the "
            + "<code>short_path</code>. The root may be empty, which it usually is for "
            + "non-generated files. For generated files it usually contains a "
            + "configuration-specific path fragment that encodes things like the target CPU "
            + "architecture that was used while building said file. Use the "
            + "<code>short_path</code> for the path under which the file is mapped if it's in the "
            + "runfiles of a binary."
  )
  public final String getExecPathString() {
    return getExecPath().getPathString();
  }

  /*
   * Returns getExecPathString escaped for potential use in a shell command.
   */
  public final String getShellEscapedExecPathString() {
    return ShellUtils.shellEscape(getExecPathString());
  }

  public final String getRootRelativePathString() {
    return getRootRelativePath().getPathString();
  }

  public final String prettyPrint() {
    // toDetailString would probably be more useful to users, but lots of tests rely on the
    // current values.
    return rootRelativePath.toString();
  }

  @Override
  public final boolean equals(Object other) {
    if (!(other instanceof Artifact)) {
      return false;
    }
    // We don't bother to check root in the equivalence relation, because we
    // assume that no root is an ancestor of another one.
    Artifact that = (Artifact) other;
    return Objects.equals(this.path, that.path);
  }

  @Override
  public final int hashCode() {
    // This is just path.hashCode().  We cache a copy in the Artifact object to reduce LLC misses
    // during operations which build a HashSet out of many Artifacts.  This is a slight loss for
    // memory but saves ~1% overall CPU in some real builds.
    return hashCode;
  }

  @Override
  public final String toString() {
    return "File:" + toDetailString();
  }

  /**
   * Returns the root-part of a given path by trimming off the end specified by
   * a given tail. Assumes that the tail is known to match, and simply relies on
   * the segment lengths.
   */
  private static PathFragment trimTail(PathFragment path, PathFragment tail) {
    return path.subFragment(0, path.segmentCount() - tail.segmentCount());
  }

  /**
   * Returns a string representing the complete artifact path information.
   */
  public final String toDetailString() {
    if (isSourceArtifact()) {
      // Source Artifact: relPath == execPath, & real path is not under execRoot
      return "[" + root + "]" + rootRelativePath;
    } else {
      // Derived Artifact: path and root are under execRoot
      PathFragment execRoot = trimTail(path.asFragment(), execPath);
      return "[[" + execRoot + "]" + root.getPath().asFragment().relativeTo(execRoot) + "]"
          + rootRelativePath;
    }
  }

  /**
   * Serializes this artifact to a string that has enough data to reconstruct the artifact.
   */
  public final String serializeToString() {
    // In theory, it should be enough to serialize execPath and rootRelativePath (which is a suffix
    // of execPath). However, in practice there is code around that uses other attributes which
    // needs cleaning up.
    String result = execPath + " /" + rootRelativePath.toString().length();
    if (getOwner() != null) {
      result += " " + getOwner();
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // Static methods to assist in working with Artifacts

  /** Formatter for execPath PathFragment output. */
  public static final Function<Artifact, String> ROOT_RELATIVE_PATH_STRING =
      artifact -> artifact.getRootRelativePath().getPathString();

  /**
   * Converts a collection of artifacts into execution-time path strings, and
   * adds those to a given collection. Middleman artifacts are ignored by this
   * method.
   */
  public static void addExecPaths(Iterable<Artifact> artifacts, Collection<String> output) {
    addNonMiddlemanArtifacts(artifacts, output, ActionInputHelper.EXEC_PATH_STRING_FORMATTER);
  }

  /**
   * Converts a collection of artifacts into the outputs computed by
   * outputFormatter and adds them to a given collection. Middleman artifacts
   * are ignored.
   */
  static <E> void addNonMiddlemanArtifacts(Iterable<Artifact> artifacts,
      Collection<? super E> output, Function<? super Artifact, E> outputFormatter) {
    for (Artifact artifact : artifacts) {
      if (MIDDLEMAN_FILTER.apply(artifact)) {
        output.add(outputFormatter.apply(artifact));
      }
    }
  }

  /**
   * Lazily converts artifacts into absolute path strings. Middleman artifacts are ignored by
   * this method.
   */
  public static Iterable<String> toAbsolutePaths(Iterable<Artifact> artifacts) {
    return Iterables.transform(
        Iterables.filter(artifacts, MIDDLEMAN_FILTER),
        artifact -> artifact.getPath().getPathString());
  }

  /**
   * Lazily converts artifacts into root-relative path strings. Middleman artifacts are ignored by
   * this method.
   */
  public static Iterable<String> toRootRelativePaths(Iterable<Artifact> artifacts) {
    return Iterables.transform(
        Iterables.filter(artifacts, MIDDLEMAN_FILTER),
        artifact -> artifact.getRootRelativePath().getPathString());
  }

  /**
   * Lazily converts artifacts into execution-time path strings. Middleman artifacts are ignored by
   * this method.
   */
  public static Iterable<String> toExecPaths(Iterable<Artifact> artifacts) {
    return ActionInputHelper.toExecPaths(Iterables.filter(artifacts, MIDDLEMAN_FILTER));
  }

  /**
   * Converts a collection of artifacts into execution-time path strings, and
   * returns those as an immutable list. Middleman artifacts are ignored by this method.
   */
  public static List<String> asExecPaths(Iterable<Artifact> artifacts) {
    return ImmutableList.copyOf(toExecPaths(artifacts));
  }

  /**
   * Renders a collection of artifacts as execution-time paths and joins
   * them into a single string. Middleman artifacts are ignored by this method.
   */
  public static String joinExecPaths(String delimiter, Iterable<Artifact> artifacts) {
    return Joiner.on(delimiter).join(toExecPaths(artifacts));
  }

  /**
   * Renders a collection of artifacts as root-relative paths and joins
   * them into a single string. Middleman artifacts are ignored by this method.
   */
  public static String joinRootRelativePaths(String delimiter, Iterable<Artifact> artifacts) {
    return Joiner.on(delimiter).join(toRootRelativePaths(artifacts));
  }

  /**
   * Adds a collection of artifacts to a given collection, with
   * {@link MiddlemanType#AGGREGATING_MIDDLEMAN} middleman actions expanded once.
   */
  public static void addExpandedArtifacts(Iterable<Artifact> artifacts,
      Collection<? super Artifact> output, ArtifactExpander artifactExpander) {
    addExpandedArtifacts(artifacts, output, Functions.<Artifact>identity(), artifactExpander);
  }

  /**
   * Converts a collection of artifacts into execution-time path strings, and
   * adds those to a given collection. Middleman artifacts for
   * {@link MiddlemanType#AGGREGATING_MIDDLEMAN} middleman actions are expanded
   * once.
   */
  @VisibleForTesting
  public static void addExpandedExecPathStrings(Iterable<Artifact> artifacts,
                                                 Collection<String> output,
                                                 ArtifactExpander artifactExpander) {
    addExpandedArtifacts(artifacts, output, ActionInputHelper.EXEC_PATH_STRING_FORMATTER,
        artifactExpander);
  }

  /**
   * Converts a collection of artifacts into execution-time path fragments, and
   * adds those to a given collection. Middleman artifacts for
   * {@link MiddlemanType#AGGREGATING_MIDDLEMAN} middleman actions are expanded
   * once.
   */
  public static void addExpandedExecPaths(Iterable<Artifact> artifacts,
      Collection<PathFragment> output, ArtifactExpander artifactExpander) {
    addExpandedArtifacts(artifacts, output, Artifact::getExecPath, artifactExpander);
  }

  /**
   * Converts a collection of artifacts into the outputs computed by
   * outputFormatter and adds them to a given collection. Middleman artifacts
   * are expanded once.
   */
  private static <E> void addExpandedArtifacts(Iterable<? extends Artifact> artifacts,
                                               Collection<? super E> output,
                                               Function<? super Artifact, E> outputFormatter,
                                               ArtifactExpander artifactExpander) {
    for (Artifact artifact : artifacts) {
      if (artifact.isMiddlemanArtifact() || artifact.isTreeArtifact()) {
        expandArtifact(artifact, output, outputFormatter, artifactExpander);
      } else {
        output.add(outputFormatter.apply(artifact));
      }
    }
  }

  private static <E> void expandArtifact(Artifact middleman,
      Collection<? super E> output,
      Function<? super Artifact, E> outputFormatter,
      ArtifactExpander artifactExpander) {
    Preconditions.checkArgument(middleman.isMiddlemanArtifact() || middleman.isTreeArtifact());
    List<Artifact> artifacts = new ArrayList<>();
    artifactExpander.expand(middleman, artifacts);
    for (Artifact artifact : artifacts) {
      output.add(outputFormatter.apply(artifact));
    }
  }

  /**
   * Converts a collection of artifacts into execution-time path strings, and
   * returns those as a list. Middleman artifacts are expanded once. The
   * returned list is mutable.
   */
  public static List<String> asExpandedExecPathStrings(Iterable<Artifact> artifacts,
                                                       ArtifactExpander artifactExpander) {
    List<String> result = new ArrayList<>();
    addExpandedExecPathStrings(artifacts, result, artifactExpander);
    return result;
  }

  /**
   * Converts a collection of artifacts into execution-time path fragments, and
   * returns those as a list. Middleman artifacts are expanded once. The
   * returned list is mutable.
   */
  public static List<PathFragment> asExpandedExecPaths(Iterable<Artifact> artifacts,
                                                       ArtifactExpander artifactExpander) {
    List<PathFragment> result = new ArrayList<>();
    addExpandedExecPaths(artifacts, result, artifactExpander);
    return result;
  }

  /**
   * Converts a collection of artifacts into execution-time path strings with
   * the root-break delimited with a colon ':', and adds those to a given list.
   * <pre>
   * Source: sourceRoot/rootRelative => :rootRelative
   * Derived: execRoot/rootPrefix/rootRelative => rootPrefix:rootRelative
   * </pre>
   */
  public static void addRootPrefixedExecPaths(Iterable<Artifact> artifacts,
      List<String> output) {
    for (Artifact artifact : artifacts) {
      output.add(asRootPrefixedExecPath(artifact));
    }
  }

  /**
   * Convenience method to filter the files to build for a certain filetype.
   *
   * @param artifacts the files to filter
   * @param allowedType the allowed filetype
   * @return all members of filesToBuild that are of one of the
   *     allowed filetypes
   */
  public static List<Artifact> filterFiles(Iterable<Artifact> artifacts, FileType allowedType) {
    List<Artifact> filesToBuild = new ArrayList<>();
    for (Artifact artifact : artifacts) {
      if (allowedType.matches(artifact.getFilename())) {
        filesToBuild.add(artifact);
      }
    }
    return filesToBuild;
  }

  @VisibleForTesting
  static String asRootPrefixedExecPath(Artifact artifact) {
    PathFragment execPath = artifact.getExecPath();
    PathFragment rootRel = artifact.getRootRelativePath();
    if (execPath.equals(rootRel)) {
      return ":" + rootRel.getPathString();
    } else { //if (execPath.endsWith(rootRel)) {
      PathFragment rootPrefix = trimTail(execPath, rootRel);
      return rootPrefix.getPathString() + ":" + rootRel.getPathString();
    }
  }

  /**
   * Converts artifacts into their exec paths. Returns an immutable list.
   */
  public static List<PathFragment> asPathFragments(Iterable<? extends Artifact> artifacts) {
    return Streams.stream(artifacts).map(Artifact::getExecPath).collect(toImmutableList());
  }

  /**
   * Returns the exec paths of the input artifacts in alphabetical order.
   */
  public static ImmutableList<PathFragment> asSortedPathFragments(Iterable<Artifact> input) {
    return Streams.stream(input).map(Artifact::getExecPath).sorted().collect(toImmutableList());
  }


  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    if (isSourceArtifact()) {
      printer.append("<source file " + rootRelativePath + ">");
    } else {
      printer.append("<generated file " + rootRelativePath + ">");
    }
  }
}
