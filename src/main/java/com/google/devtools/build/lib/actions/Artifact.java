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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.actions.ArtifactResolver.ArtifactResolverSupplier;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.EvalUtils.ComparisonException;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.ShareabilityOfValue;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * An Artifact represents a file used by the build system, whether it's a source file or a derived
 * (output) file. Not all Artifacts have a corresponding FileTarget object in the <code>
 * build.lib.packages</code> API: for example, low-level intermediaries internal to a given rule,
 * such as a Java class files or C++ object files. However all FileTargets have a corresponding
 * Artifact.
 *
 * <p>In any given call to SkyframeExecutor#buildArtifacts(), no two Artifacts in the action graph
 * may refer to the same path.
 *
 * <p>Artifacts generally fall into two classifications, source and derived, but there exist a few
 * other cases that are fuzzy and difficult to classify. The following cases exist:
 *
 * <ul>
 *   <li>Well-formed source Artifacts will have null generating Actions and a root that is
 *       orthogonal to execRoot. (With the root coming from the package path.)
 *   <li>Well-formed derived Artifacts will have non-null generating Actions, and a root that is
 *       below execRoot.
 *   <li>Symlinked include source Artifacts under the output/include tree will appear to be derived
 *       artifacts with null generating Actions.
 *   <li>Some derived Artifacts, mostly in the genfiles tree and mostly discovered during include
 *       validation, will also have null generating Actions.
 * </ul>
 *
 * In the usual case, an Artifact represents a single file. However, an Artifact may also represent
 * the following:
 *
 * <ul>
 *   <li>A TreeArtifact, which is a directory containing a tree of unknown {@link Artifact}s. In the
 *       future, Actions will be able to examine these files as inputs and declare them as outputs
 *       at execution time, but this is not yet implemented. This is used for Actions where the
 *       inputs and/or outputs might not be discoverable except during Action execution.
 *   <li>A directory of unknown contents, but not a TreeArtifact. This is a legacy facility and
 *       should not be used by any new rule implementations. In particular, the file system cache
 *       integrity checks fail for directories.
 *   <li>An 'aggregating middleman' special Artifact, which may be expanded using a {@link
 *       ArtifactExpander} at Action execution time. This is used by a handful of rules to save
 *       memory.
 *   <li>A 'constant metadata' special Artifact. These represent real files, changes to which are
 *       ignored by the build system. They are useful for files which change frequently but do not
 *       affect the result of a build, such as timestamp files.
 *   <li>A 'Fileset' special Artifact. This is a legacy type of Artifact and should not be used by
 *       new rule implementations.
 * </ul>
 */
@Immutable
@AutoCodec
public class Artifact
    implements FileType.HasFileType,
        ActionInput,
        FileApi,
        Comparable<Object>,
        CommandLineItem,
        SkyKey {

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

  /**
   * {@link com.google.devtools.build.lib.skyframe.ArtifactFunction} does direct filesystem access
   * without declaring Skyframe dependencies if the artifact is a source directory. However, that
   * filesystem access is not invalidated on incremental builds, and we have no plans to fix it,
   * since general consumption of source directories in this way is unsound. Therefore no new bugs
   * are created by declaring {@link com.google.devtools.build.lib.skyframe.ArtifactFunction} to be
   * hermetic.
   *
   * <p>TODO(janakr): Avoid this issue entirely by giving {@link SourceArtifact} its own {@code
   * SkyFunction}. Then we can just declare that function to be non-hermetic. That will also save
   * memory since we can make mandatory source artifacts their own SkyKeys!
   */
  public static final SkyFunctionName ARTIFACT = SkyFunctionName.createHermetic("ARTIFACT");

  @Override
  public int compareTo(Object o) {
    if (o instanceof Artifact) {
      return EXEC_PATH_COMPARATOR.compare(this, (Artifact) o);
    }
    throw new ComparisonException("Cannot compare artifact with " + EvalUtils.getDataTypeName(o));
  }

  /** An object that can expand middleman and tree artifacts. */
  public interface ArtifactExpander {

    /**
     * Expands the given artifact, and populates "output" with the result.
     *
     * <p>{@code artifact.isMiddlemanArtifact() || artifact.isTreeArtifact()} must be true.
     * Only aggregating middlemen and tree artifacts are expanded.
     */
    void expand(Artifact artifact, Collection<? super Artifact> output);

    /**
     * Retrieve the expansion of Filesets for the given artifact.
     *
     * @param artifact {@code artifact.isFileset()} must be true.
     */
    default ImmutableList<FilesetOutputSymlink> getFileset(Artifact artifact) {
      throw new UnsupportedOperationException();
    }
  }

  /** Implementation of {@link ArtifactExpander} */
  public static class ArtifactExpanderImpl implements ArtifactExpander {
    private final Map<Artifact, Collection<Artifact>> expandedInputs;
    private final Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets;

    public ArtifactExpanderImpl(
        Map<Artifact, Collection<Artifact>> expandedInputs,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets) {
      this.expandedInputs = expandedInputs;
      this.expandedFilesets = expandedFilesets;
    }

    @Override
    public void expand(Artifact artifact, Collection<? super Artifact> output) {
      Preconditions.checkState(
          artifact.isMiddlemanArtifact() || artifact.isTreeArtifact(), artifact);
      Collection<Artifact> result = expandedInputs.get(artifact);
      if (result != null) {
        output.addAll(result);
      }
    }

    @Override
    public ImmutableList<FilesetOutputSymlink> getFileset(Artifact artifact) {
      Preconditions.checkState(artifact.isFileset());
      return Preconditions.checkNotNull(expandedFilesets.get(artifact));
    }
  }

  public static final ImmutableList<Artifact> NO_ARTIFACTS = ImmutableList.of();

  /** A Predicate that evaluates to true if the Artifact is not a middleman artifact. */
  public static final Predicate<Artifact> MIDDLEMAN_FILTER = input -> !input.isMiddlemanArtifact();

  private static final Interner<Artifact> ARTIFACT_INTERNER = BlazeInterners.newWeakInterner();

  private final int hashCode;
  private final ArtifactRoot root;
  private final PathFragment execPath;
  private final PathFragment rootRelativePath;
  private final ArtifactOwner owner;

  /**
   * The {@code rootRelativePath is a few characters shorter than the {@code execPath} for derived
   * artifacts, so we save a few bytes by serializing it rather than the {@code execPath},
   * especially when the {@code root} is common to many artifacts and therefore memoized.
   */
  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static Artifact createForSerialization(
      ArtifactRoot root, ArtifactOwner owner, PathFragment rootRelativePath) {
    if (rootRelativePath == null || rootRelativePath.isAbsolute() != root.getRoot().isAbsolute()) {
      throw new IllegalArgumentException(
          rootRelativePath
              + ": illegal rootRelativePath for "
              + rootRelativePath
              + " (root: "
              + root
              + ")");
    }
    PathFragment rootExecPath = root.getExecPath();
    Artifact artifact = new Artifact(
        root,
        rootExecPath.isEmpty() ? rootRelativePath : rootExecPath.getRelative(rootRelativePath),
        rootRelativePath,
        owner);
    if (artifact.isSourceArtifact()) {
      return artifact;
    } else {
      return ARTIFACT_INTERNER.intern(artifact);
    }
  }

  /**
   * Constructs an artifact for the specified path, root and execPath. The root must be an ancestor
   * of path, and execPath must be a non-absolute tail of path. Outside of testing, this method
   * should only be called by ArtifactFactory. The ArtifactOwner may be null.
   *
   * <p>In a source Artifact, the path tail after the root will be identical to the execPath, but
   * the root will be orthogonal to execRoot.
   *
   * <pre>
   *  [path] == [/root/][execPath]
   * </pre>
   *
   * <p>In a derived Artifact, the execPath will overlap with part of the root, which in turn will
   * be below the execRoot.
   *
   * <pre>
   *  [path] == [/root][pathTail] == [/execRoot][execPath] == [/execRoot][rootPrefix][pathTail]
   * </pre>
   */
  @VisibleForTesting
  public Artifact(ArtifactRoot root, PathFragment execPath, ArtifactOwner owner) {
    this(
        Preconditions.checkNotNull(root),
        Preconditions.checkNotNull(execPath, "Null execPath not allowed (root %s", root),
        root.getExecPath().isEmpty() ? execPath : execPath.relativeTo(root.getExecPath()),
        Preconditions.checkNotNull(owner));
    if (execPath.isAbsolute() != root.getRoot().isAbsolute()) {
      throw new IllegalArgumentException(
          execPath + ": illegal execPath for " + execPath + " (root: " + root + ")");
    }
  }

  private Artifact(
      ArtifactRoot root,
      PathFragment execPath,
      PathFragment rootRelativePath,
      ArtifactOwner owner) {
    Preconditions.checkNotNull(root);
    if (execPath.isEmpty()) {
      throw new IllegalArgumentException(
          "it is illegal to create an artifact with an empty execPath");
    }
    // The ArtifactOwner is not part of this computation because it is very rare that two Artifacts
    // have the same execPath and different owners, so a collision is fine there. If this is
    // changed, OwnerlessArtifactWrapper must also be changed.
    this.hashCode = execPath.hashCode();
    this.root = root;
    this.execPath = execPath;
    this.rootRelativePath = rootRelativePath;
    this.owner = Preconditions.checkNotNull(owner);
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
  public Artifact(ArtifactRoot root, PathFragment execPath) {
    this(root, execPath, ArtifactOwner.NullArtifactOwner.INSTANCE);
  }

  /**
   * Constructs a source or derived Artifact for the specified path and specified root. The root
   * must be an ancestor of the path.
   */
  @VisibleForTesting // Only exists for testing.
  public Artifact(Path path, ArtifactRoot root) {
    this(
        root,
        root.getExecPath().getRelative(root.getRoot().relativize(path)),
        ArtifactOwner.NullArtifactOwner.INSTANCE);
  }

  /** Constructs a source or derived Artifact for the specified root-relative path and root. */
  @VisibleForTesting // Only exists for testing.
  public Artifact(PathFragment rootRelativePath, ArtifactRoot root) {
    this(
        root,
        root.getExecPath().getRelative(rootRelativePath),
        ArtifactOwner.NullArtifactOwner.INSTANCE);
  }

  public final Path getPath() {
    return root.getRoot().getRelative(rootRelativePath);
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
  @Override
  public final String getDirname() {
    PathFragment parent = getExecPath().getParentDirectory();
    return (parent == null) ? "/" : parent.getSafePathString();
  }

  /**
   * Returns the base file name of this artifact, similar to basename(1).
   */
  @Override
  public final String getFilename() {
    return getExecPath().getBaseName();
  }

  @Override
  public final String getExtension() {
    return getExecPath().getFileExtension();
  }

  /**
   * Checks whether this artifact is of the supplied file type.
   *
   * <p>Prefer this method to pulling out strings from the Artifact and passing to {@link
   * FileType#matches(String)} manually. This method has been optimized to generate a minimum of
   * garbage.
   */
  public boolean isFileType(FileType fileType) {
    return fileType.matches(this);
  }

  @Override
  public String filePathForFileTypeMatcher() {
    return getExecPath().filePathForFileTypeMatcher();
  }

  @Override
  public String expandToCommandLine() {
    return getExecPathString();
  }

  /**
   * Returns the artifact owner. May be null.
   */
  @Nullable public final Label getOwner() {
    return owner.getLabel();
  }

  /**
   * Gets the {@code ActionLookupKey} of the {@code ConfiguredTarget} that owns this artifact, if it
   * was set. Otherwise, this should be a dummy value -- either {@link
   * ArtifactOwner.NullArtifactOwner#INSTANCE} or a dummy owner set in tests. Such a dummy value
   * should only occur for source artifacts if created without specifying the owner, or for special
   * derived artifacts, such as target completion middleman artifacts, build info artifacts, and the
   * like.
   */
  public final ArtifactOwner getArtifactOwner() {
    return owner;
  }

  @Override
  public Label getOwnerLabel() {
    return owner.getLabel();
  }

  /**
   * Returns the root beneath which this Artifact resides, if any. This may be one of the
   * package-path entries (for source Artifacts), or one of the bin, genfiles or includes dirs (for
   * derived Artifacts). It will always be an ancestor of getPath().
   */
  @Override
  public final ArtifactRoot getRoot() {
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
  @Override
  public final boolean isSourceArtifact() {
    return root.isSourceRoot();
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
  public boolean isTreeArtifact() {
    return false;
  }

  /**
   * Returns whether the artifact represents a Fileset.
   */
  public boolean isFileset() {
    return false;
  }

  @Override
  public boolean isDirectory() {
    return isTreeArtifact() || isFileset();
  }

  /**
   * Returns true iff metadata cache must return constant metadata for the
   * given artifact.
   */
  public boolean isConstantMetadata() {
    return false;
  }

  /** {@link Artifact#isSourceArtifact() is true.
   *
   * <p>Source artifacts have the property that unlike for output artifacts, direct file system
   * access for their contents should be safe, even in a distributed context.
   *
   * TODO(shahan): move {@link Artifact#getPath} to this subclass.
   * */
  public static final class SourceArtifact extends Artifact {
    @VisibleForTesting
    public SourceArtifact(ArtifactRoot root, PathFragment execPath, ArtifactOwner owner) {
      super(root, execPath, owner);
    }

    /**
     * SourceArtifacts are compared without their owners, since owners do not affect behavior,
     * unlike with derived artifacts, whose owners determine their generating actions.
     */
    @Override
    public boolean equals(Object other) {
      return other instanceof SourceArtifact && equalsWithoutOwner((SourceArtifact) other);
    }
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
   * <p>We subclass {@link Artifact} instead of storing the special attributes inside in order to
   * save memory. The proportion of artifacts that are special is very small, and by not having to
   * keep around the attribute for the rest we save some memory.
   */
  @Immutable
  @VisibleForTesting
  @AutoCodec
  public static final class SpecialArtifact extends Artifact {
    private final SpecialArtifactType type;

    @VisibleForSerialization
    public SpecialArtifact(
        ArtifactRoot root, PathFragment execPath, ArtifactOwner owner, SpecialArtifactType type) {
      super(root, execPath, owner);
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

    @Override
    public ShareabilityOfValue getShareabilityOfValue() {
      return isConstantMetadata() ? ShareabilityOfValue.NEVER : super.getShareabilityOfValue();
    }
  }

  /**
   * A special kind of artifact that represents a concrete file created at execution time under its
   * associated TreeArtifact.
   *
   * <p>TreeFileArtifacts should be only created during execution time inside some special actions
   * to support action inputs and outputs that are unpredictable at analysis time. TreeFileArtifacts
   * should not be created directly by any rules at analysis time.
   *
   * <p>We subclass {@link Artifact} instead of storing the extra fields directly inside in order to
   * save memory. The proportion of TreeFileArtifacts is very small, and by not having to keep
   * around the extra fields for the rest we save some memory.
   */
  @Immutable
  @AutoCodec
  public static final class TreeFileArtifact extends Artifact {
    private final SpecialArtifact parentTreeArtifact;
    private final PathFragment parentRelativePath;

    /**
     * Constructs a TreeFileArtifact with the given parent-relative path under the given parent
     * TreeArtifact. The {@link ArtifactOwner} of the TreeFileArtifact is the {@link ArtifactOwner}
     * of the parent TreeArtifact.
     */
    @VisibleForTesting
    public TreeFileArtifact(SpecialArtifact parent, PathFragment parentRelativePath) {
      this(parent, parentRelativePath, parent.getArtifactOwner());
    }

    /**
     * Constructs a TreeFileArtifact with the given parent-relative path under the given parent
     * TreeArtifact, owned by the given {@code artifactOwner}.
     */
    @AutoCodec.Instantiator
    TreeFileArtifact(
        SpecialArtifact parentTreeArtifact, PathFragment parentRelativePath, ArtifactOwner owner) {
      super(
          parentTreeArtifact.getRoot(),
          parentTreeArtifact.getExecPath().getRelative(parentRelativePath),
          owner);
      Preconditions.checkArgument(
          parentTreeArtifact.isTreeArtifact(),
          "The parent of TreeFileArtifact (parent-relative path: %s) is not a TreeArtifact: %s",
          parentRelativePath,
          parentTreeArtifact);
      Preconditions.checkArgument(
          !parentRelativePath.containsUplevelReferences() && !parentRelativePath.isAbsolute(),
          "%s is not a proper normalized relative path",
          parentRelativePath);
      this.parentTreeArtifact = parentTreeArtifact;
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

  @Override
  public final String getRunfilesPathString() {
    return getRunfilesPath().getPathString();
  }

  /**
   * Returns this.getExecPath().getPathString().
   */
  @Override
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

  @SuppressWarnings("EqualsGetClass") // Distinct classes of Artifact are never equal.
  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof Artifact)) {
      return false;
    }
    if (!getClass().equals(other.getClass())) {
      return false;
    }
    Artifact that = (Artifact) other;
    return equalsWithoutOwner(that) && owner.equals(that.owner);
  }

  public boolean equalsWithoutOwner(Artifact other) {
    return hashCode == other.hashCode && execPath.equals(other.execPath) && root.equals(other.root);
  }

  @Override
  public final int hashCode() {
    // This is just execPath.hashCode() (along with the class). We cache a copy in the Artifact
    // object to reduce LLC misses during operations which build a HashSet out of many Artifacts.
    // This is a slight loss for memory but saves ~1% overall CPU in some real builds.
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
      //
      // TODO(blaze-team): this is misleading beacuse execution_root isn't unique. Dig the
      // workspace name out and print that also.
      return "[[<execution_root>]" + root.getExecPath() + "]" + rootRelativePath;
    }
  }

  /** {@link ObjectCodec} for {@link SourceArtifact} */
  private static class SourceArtifactCodec implements ObjectCodec<SourceArtifact> {

    @Override
    public Class<? extends SourceArtifact> getEncodedClass() {
      return SourceArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, SourceArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getExecPath(), codedOut);
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getArtifactOwner(), codedOut);
    }

    @Override
    public SourceArtifact deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      PathFragment execPath = context.deserialize(codedIn);
      ArtifactRoot artifactRoot = context.deserialize(codedIn);
      ArtifactOwner owner = context.deserialize(codedIn);
      return (SourceArtifact)
          context
              .getDependency(ArtifactResolverSupplier.class)
              .get()
              .getSourceArtifact(execPath, artifactRoot.getRoot(), owner);
    }
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

  /**
   * A utility class that compares {@link Artifact}s without taking their owners into account.
   * Should only be used for detecting action conflicts and merging shared action data.
   */
  public static class OwnerlessArtifactWrapper {
    private final Artifact artifact;

    public OwnerlessArtifactWrapper(Artifact artifact) {
      this.artifact = artifact;
    }

    @Override
    public int hashCode() {
      // Depends on the fact that Artifact#hashCode does not use ArtifactOwner.
      return artifact.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof OwnerlessArtifactWrapper
          && this.artifact.equalsWithoutOwner(((OwnerlessArtifactWrapper) obj).artifact);
    }
  }

  @Override
  public SkyFunctionName functionName() {
    return ARTIFACT;
  }
}
