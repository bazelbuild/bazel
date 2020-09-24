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
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ArtifactResolver.ArtifactResolverSupplier;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
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
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;

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
 *
 * <p>While Artifact implements {@link SkyKey} for memory-saving purposes, Skyframe requests
 * involving artifacts should always go through {@link Artifact#key} since ordinary derived
 * artifacts should not be requested directly from Skyframe.
 */
@Immutable
public abstract class Artifact
    implements FileType.HasFileType,
        ActionInput,
        FileApi,
        Comparable<Artifact>,
        CommandLineItem,
        SkyKey {

  public static final Depset.ElementType TYPE = Depset.ElementType.of(Artifact.class);

  /** Compares artifact according to their exec paths. Sorts null values first. */
  @SuppressWarnings("ReferenceEquality") // "a == b" is an optimization
  public static final Comparator<Artifact> EXEC_PATH_COMPARATOR =
      (a, b) -> {
        if (a == b) {
          return 0;
        } else if (a == null) {
          return -1;
        } else if (b == null) {
          return 1;
        } else {
          return a.execPath.compareTo(b.execPath);
        }
      };

  /** Compares artifact according to their root relative paths. Sorts null values first. */
  @SuppressWarnings("ReferenceEquality") // "a == b" is an optimization
  public static final Comparator<Artifact> ROOT_RELATIVE_PATH_COMPARATOR =
      (a, b) -> {
        if (a == b) {
          return 0;
        } else if (a == null) {
          return -1;
        } else if (b == null) {
          return 1;
        } else {
          int result = a.getRootRelativePath().compareTo(b.getRootRelativePath());
          if (result == 0) {
            // Use the full exec path as a fallback if the root-relative paths are the same, thus
            // avoiding problems when ImmutableSortedMaps are switched from EXEC_PATH_COMPARATOR.
            return a.execPath.compareTo(b.execPath);
          } else {
            return result;
          }
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

  /**
   * Returns a {@link SkyKey} that, when built, will produce this artifact. For source artifacts and
   * generated artifacts that may aggregate other artifacts (middleman, since they may be
   * aggregating middlemen, and tree), returns the artifact itself. For normal generated artifacts,
   * returns the key of the generating action.
   *
   * <p>Callers should use this method (or the related ones below) in preference to directly
   * requesting an {@link Artifact} to be built by Skyframe, since ordinary derived artifacts should
   * never be directly built by Skyframe.
   */
  @ThreadSafety.ThreadSafe
  public static SkyKey key(Artifact artifact) {
    if (artifact.isTreeArtifact()
        || artifact.isMiddlemanArtifact()
        || !artifact.hasKnownGeneratingAction()) {
      return artifact;
    }

    return ((DerivedArtifact) artifact).getGeneratingActionKey();
  }

  public static Iterable<SkyKey> keys(Iterable<Artifact> artifacts) {
    return Iterables.transform(artifacts, Artifact::key);
  }

  @Override
  public int compareTo(Artifact o) {
    return EXEC_PATH_COMPARATOR.compare(this, o);
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
     * Returns the expansion of Fileset for the given artifact.
     *
     * @param artifact {@code artifact.isFileset()} must be true.
     * @throws MissingExpansionException if the expander is missing data needed to expand provided
     *     fileset.
     */
    default ImmutableList<FilesetOutputSymlink> getFileset(Artifact artifact)
        throws MissingExpansionException {
      throw new MissingExpansionException("Cannot expand fileset " + artifact);
    }

    /**
     * Return an {@link ArchivedTreeArtifact} for a provided {@linkplain SpecialArtifact tree
     * artifact} if one is available.
     *
     * <p>The {@linkplain ArchivedTreeArtifact archived tree artifact} can be used instead of the
     * tree artifact expansion.
     */
    @Nullable
    default ArchivedTreeArtifact getArchivedTreeArtifact(SpecialArtifact treeArtifact) {
      return null;
    }
  }

  /**
   * Exception thrown when attempting to {@linkplain ArtifactExpander expand} an artifact for which
   * we do not have the necessary data.
   */
  public static final class MissingExpansionException extends Exception {

    public MissingExpansionException(String message) {
      super(message);
    }
  }

  /** Implementation of {@link ArtifactExpander} */
  public static class ArtifactExpanderImpl implements ArtifactExpander {
    private final Map<Artifact, Collection<Artifact>> expandedInputs;
    private final Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts;
    private final Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets;

    public ArtifactExpanderImpl(
        Map<Artifact, Collection<Artifact>> expandedInputs,
        Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets) {
      this.expandedInputs = expandedInputs;
      this.archivedTreeArtifacts = archivedTreeArtifacts;
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
    public ImmutableList<FilesetOutputSymlink> getFileset(Artifact artifact)
        throws MissingExpansionException {
      Preconditions.checkState(artifact.isFileset());
      ImmutableList<FilesetOutputSymlink> filesetLinks = expandedFilesets.get(artifact);
      if (filesetLinks == null) {
        throw new MissingExpansionException("Missing expansion for fileset: " + artifact);
      }
      return filesetLinks;
    }

    @Override
    public ArchivedTreeArtifact getArchivedTreeArtifact(SpecialArtifact treeArtifact) {
      return archivedTreeArtifacts.get(treeArtifact);
    }
  }

  /** A Predicate that evaluates to true if the Artifact is not a middleman artifact. */
  public static final Predicate<Artifact> MIDDLEMAN_FILTER = input -> !input.isMiddlemanArtifact();

  protected final ArtifactRoot root;

  private final int hashCode;
  private final PathFragment execPath;

  private Artifact(ArtifactRoot root, PathFragment execPath) {
    Preconditions.checkNotNull(root);
    // The ArtifactOwner is not part of this computation because it is very rare that two Artifacts
    // have the same execPath and different owners, so a collision is fine there. If this is
    // changed, OwnerlessArtifactWrapper must also be changed.
    this.hashCode = execPath.hashCode();
    this.root = root;
    this.execPath = execPath;
  }

  /**
   * An artifact corresponding to a file in the output tree, generated by an {@link Action}, but for
   * which the generating action is unknown (may have been created as an undeclared "side effect" of
   * another action).
   *
   * <p>This artifact type defined for compatibility with the Ninja build system.
   *
   * <p>Only use this type in rare cases, and only when there is certainty that the artifact is
   * actually generated by an action in the build (for example, when another action declares it as
   * an explicit dependency in a post-execution .d file). These artifacts are indicative of
   * underspecified, incomplete builds, and may in worst cases be indicative of incorrect builds.
   */
  @Immutable
  public static final class NinjaMysteryArtifact extends Artifact {
    public NinjaMysteryArtifact(ArtifactRoot root, PathFragment execPath) {
      super(root, execPath);
    }

    @Override
    public PathFragment getRootRelativePath() {
      return getExecPath().relativeTo(getRoot().getExecPath());
    }

    @Override
    boolean ownersEqual(Artifact other) {
      return true;
    }

    @Override
    public ArtifactOwner getArtifactOwner() {
      return ArtifactOwner.NULL_OWNER;
    }

    @Override
    public Label getOwnerLabel() {
      return ArtifactOwner.NULL_OWNER.getLabel();
    }

    @Override
    public boolean hasKnownGeneratingAction() {
      return false;
    }
  }

  /** An artifact corresponding to a file in the output tree, generated by an {@link Action}. */
  public static class DerivedArtifact extends Artifact {

    /**
     * An {@link ActionLookupKey} until {@link #setGeneratingActionKey} is set, at which point it is
     * an {@link ActionLookupData}, whose {@link ActionLookupData#getActionLookupKey} will be the
     * same as the original value of owner.
     *
     * <p>We overload this field in order to save memory.
     */
    private Object owner;

    /**
     * Content-based output paths are experimental. Only derived artifacts that are explicitly opted
     * in by their creating rules should use them and only when {@link
     * com.google.devtools.build.lib.analysis.config.BuildConfiguration#useContentBasedOutputPaths}
     * is on.
     */
    private final boolean contentBasedPath;

    /** Standard constructor for derived artifacts. */
    public DerivedArtifact(ArtifactRoot root, PathFragment execPath, ActionLookupKey owner) {
      this(root, execPath, owner, /*contentBasedPath=*/ false);
    }

    /**
     * Same as {@link #DerivedArtifact(ArtifactRoot, PathFragment, ActionLookupKey)} but includes
     * tge option to use a content-based path for this artifact (see {@link
     * com.google.devtools.build.lib.analysis.config.BuildConfiguration#useContentBasedOutputPaths}).
     */
    public DerivedArtifact(
        ArtifactRoot root, PathFragment execPath, ActionLookupKey owner, boolean contentBasedPath) {
      super(root, execPath);
      Preconditions.checkState(
          !root.getExecPath().isEmpty(), "Derived root has no exec path: %s, %s", root, execPath);
      this.owner = owner;
      this.contentBasedPath = contentBasedPath;
    }

    /**
     * Called when a configured target's actions are being collected. {@code generatingActionKey}
     * must have the same owner as this artifact's current {@link #getArtifactOwner}.
     */
    @VisibleForTesting
    public final void setGeneratingActionKey(ActionLookupData generatingActionKey) {
      Preconditions.checkState(
          this.owner instanceof ArtifactOwner,
          "Already set generating action key: %s (%s %s)",
          this,
          this.owner,
          generatingActionKey);
      Preconditions.checkState(
          Preconditions.checkNotNull(generatingActionKey, this).getActionLookupKey().equals(owner),
          "Owner of generating action key not same as artifact's owner: %s (%s %s)",
          this,
          this.owner,
          generatingActionKey);
      this.owner = Preconditions.checkNotNull(generatingActionKey, this);
    }

    @VisibleForTesting
    public final boolean hasGeneratingActionKey() {
      return this.owner instanceof ActionLookupData;
    }

    /** Can only be called once {@link #setGeneratingActionKey} is called. */
    public final ActionLookupData getGeneratingActionKey() {
      Preconditions.checkState(owner instanceof ActionLookupData, "Bad owner: %s %s", this, owner);
      return (ActionLookupData) owner;
    }

    @Override
    public final ActionLookupKey getArtifactOwner() {
      return owner instanceof ActionLookupData
          ? getGeneratingActionKey().getActionLookupKey()
          : (ActionLookupKey) owner;
    }

    @Override
    public final Label getOwnerLabel() {
      return getArtifactOwner().getLabel();
    }

    @Override
    public final PathFragment getRootRelativePath() {
      return getExecPath().relativeTo(getRoot().getExecPath());
    }

    @Override
    final boolean ownersEqual(Artifact other) {
      DerivedArtifact that = (DerivedArtifact) other;
      if (!(this.owner instanceof ActionLookupData) || !(that.owner instanceof ActionLookupData)) {
        // Happens when at least one of these artifacts hasn't had its generating action key set
        // yet, so its configured target is still being analyzed. Tolerate.
        return this.getArtifactOwner().equals(that.getArtifactOwner());
      }
      return this.owner.equals(that.owner);
    }

    @Override
    public boolean contentBasedPath() {
      return contentBasedPath;
    }
  }

  @SuppressWarnings("unused") // Codec used by reflection.
  private static class DerivedArtifactCodec implements ObjectCodec<DerivedArtifact> {
    @Override
    public Class<DerivedArtifact> getEncodedClass() {
      return DerivedArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, DerivedArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getGeneratingActionKey(), codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
    }

    @Override
    public DerivedArtifact deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ArtifactRoot root = context.deserialize(codedIn);
      ActionLookupData generatingActionKey = context.deserialize(codedIn);
      DerivedArtifact artifact =
          new DerivedArtifact(
              root,
              validateAndGetRootExecPath(root, generatingActionKey, context, codedIn),
              generatingActionKey.getActionLookupKey(),
              /*contentBasedPath=*/ false);
      artifact.setGeneratingActionKey(generatingActionKey);
      return context
          .getDependency(ArtifactResolver.ArtifactResolverSupplier.class)
          .intern(artifact);
    }

    static PathFragment validateAndGetRootExecPath(
        ArtifactRoot root,
        ActionLookupData generatingActionKey,
        DeserializationContext context,
        CodedInputStream codedIn)
        throws IOException, SerializationException {
      PathFragment rootRelativePath = context.deserialize(codedIn);
      if (rootRelativePath == null
          || rootRelativePath.isAbsolute() != root.getRoot().isAbsolute()) {
        throw new IllegalArgumentException(
            rootRelativePath
                + ": illegal rootRelativePath for "
                + root
                + " (generatingActionKey: "
                + generatingActionKey
                + ")");
      }
      Preconditions.checkState(
          !root.isSourceRoot(), "Root not derived: %s %s", root, rootRelativePath);
      return root.getExecPath().getRelative(rootRelativePath);
    }
  }

  public final Path getPath() {
    return root.getRoot().getRelative(getRootRelativePath());
  }

  public boolean hasParent() {
    return getParent() != null;
  }

  /**
   * Returns the parent Artifact containing this Artifact. Artifacts without parents shall return
   * null.
   */
  @Nullable
  public SpecialArtifact getParent() {
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

  /** Checks whether this artifact is of one of the types in the supplied set. */
  public boolean isFileType(FileTypeSet fileTypeSet) {
    return fileTypeSet.matches(filePathForFileTypeMatcher());
  }

  @Override
  public final String filePathForFileTypeMatcher() {
    return getExecPath().filePathForFileTypeMatcher();
  }

  @Override
  public final String expandToCommandLine() {
    return getExecPathString();
  }

  /** Returns the artifact's owning label. May be null. */
  @Nullable
  public final Label getOwner() {
    return getOwnerLabel();
  }

  /**
   * Gets the {@code ActionLookupKey} of the {@code ConfiguredTarget} that owns this artifact, if it
   * was set. Otherwise, this should be a dummy value -- either {@link ArtifactOwner#NULL_OWNER} or
   * a dummy owner set in tests. Such a dummy value should only occur for source artifacts if
   * created without specifying the owner, or for special derived artifacts, such as target
   * completion middleman artifacts, build info artifacts, and the like.
   */
  public abstract ArtifactOwner getArtifactOwner();

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
   * Returns the relative path to this artifact relative to its root. It makes no guarantees as to
   * the semantic meaning or the completeness of the returned path value. In other words, no
   * assumptions should be made in terms of where the root portion of the path ends, and the
   * returned value almost always needs to be used in conjunction with its root.
   *
   * <p>{#link Artifact#getOutputDirRelativePath()} is more versatile for general use cases.
   */
  public abstract PathFragment getRootRelativePath();

  /**
   * Returns the fully-qualified package path to this artifact. By "fully-qualified", it means the
   * returned path is prefixed with "external/<repository name>" if this artifact is in an external
   * repository.
   *
   * <p>Do not call this method just because you need a path prefixed with the "external/<repository
   * name>" fragment for external repository artifacts. {@link * Artifact#getOutputDirRelativePath}
   * is the right one to use in almost all cases.
   *
   * @deprecated This method is only to be used for $(location) and getOutputDirRelativePath
   *     implementations.
   */
  @Deprecated
  public PathFragment getPathForLocationExpansion() {
    return getRootRelativePath();
  }

  /**
   * Returns the path to this artifact relative to an output directory, e.g. the bin directory. Note
   * that this is available on every Artifact type, including source artifacts. As a matter of fact,
   * one of its most common use cases is to construct a derived artifact's output path out of a
   * sibling source artifact's by replacing the basename in its output-dir-relative path.
   *
   * <p>This is just a wrapper over {@link Artifact#getPathForLocationExpansion} at the moment.
   * However, since it will be kept in sync with the output directory layout, which is planned for
   * an overhaul, it must be preferred over {@link Artifact#getPathForLocationExpansion} whenever
   * possible.
   */
  public final PathFragment getOutputDirRelativePath() {
    return getPathForLocationExpansion();
  }

  /** Returns this.getExecPath().getPathString(). */
  @Override
  public final String getExecPathString() {
    return getExecPath().getPathString();
  }

  public final String getRootRelativePathString() {
    return getRootRelativePath().getPathString();
  }

  public final String getOutputDirRelativePathString() {
    return getOutputDirRelativePath().getPathString();
  }

  @Override
  public boolean contentBasedPath() {
    return false;
  }

  @Override
  public boolean isSymlink() {
    return false;
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

  @Override
  public String getTreeRelativePathString() throws EvalException {
    throw Starlark.errorf(
        "tree_relative_path not allowed for files that are not tree artifact files.");
  }

  /**
   * Returns true iff this is a source Artifact as determined by its path and root relationships.
   * Note that this will report all Artifacts in the output tree, including in the include symlink
   * tree, as non-source.
   *
   * <p>An {@link Artifact} is a {@link SourceArtifact} iff this returns true, and a {@link
   * DerivedArtifact} otherwise.
   */
  @Override
  public final boolean isSourceArtifact() {
    return root.isSourceRoot();
  }

  /**
   * Returns true iff this artifact has a generating action, and that generating action is known.
   */
  public boolean hasKnownGeneratingAction() {
    return !isSourceArtifact();
  }

  /**
   * Returns true iff this is a middleman Artifact as determined by its root.
   *
   * <p>If true, this artifact is necessarily a {@link DerivedArtifact}.
   */
  public final boolean isMiddlemanArtifact() {
    return getRoot().isMiddlemanRoot();
  }

  /**
   * Returns true iff this is a TreeArtifact representing a directory tree containing Artifacts.
   *
   * <p>if true, this artifact is necessarily a {@link SpecialArtifact} with type {@link
   * SpecialArtifactType#TREE}.
   */
  public boolean isTreeArtifact() {
    return false;
  }

  /**
   * Returns {@code true} if this is a {@link TreeFileArtifact} that was created by an action which
   * declared an output directory, as opposed to an action that was generated by an action template
   * expansion.
   *
   * <p>Such artifacts should always be stored within a {@link
   * com.google.devtools.build.lib.skyframe.TreeArtifactValue} representing the declared directory
   * and all children, not individually like other derived artifacts.
   */
  public boolean isChildOfDeclaredDirectory() {
    return false;
  }

  /**
   * Returns whether the artifact represents a Fileset.
   *
   * <p>if true, this artifact is necessarily a {@link SpecialArtifact} with type {@link
   * SpecialArtifactType#FILESET}.
   */
  public boolean isFileset() {
    return false;
  }

  /** The disjunction of {@link #isTreeArtifact} and {@link #isFileset}. */
  @Override
  public boolean isDirectory() {
    return isTreeArtifact() || isFileset();
  }

  /**
   * Returns true iff metadata cache must return constant metadata for the given artifact.
   *
   * <p>If true, this artifact is necessarily a {@link SpecialArtifact} with type {@link
   * SpecialArtifactType#CONSTANT_METADATA}.
   */
  public boolean isConstantMetadata() {
    return false;
  }

  /**
   * For targets in external repositories, this returns the path the artifact live at in the
   * runfiles tree. For local targets, it returns the rootRelativePath.
   */
  public final PathFragment getRunfilesPath() {
    PathFragment relativePath = getOutputDirRelativePath();
    // We can't use root.isExternalSource() here since it needs to handle derived artifacts too.
    if (relativePath.startsWith(LabelConstants.EXTERNAL_PATH_PREFIX)) {
      // Turn external/repo/foo into ../repo/foo.
      relativePath = relativePath.relativeTo(LabelConstants.EXTERNAL_PATH_PREFIX);
      relativePath = PathFragment.create("..").getRelative(relativePath);
    }
    return relativePath;
  }

  @Override
  public final String getRunfilesPathString() {
    return getRunfilesPath().getPathString();
  }

  public final String prettyPrint() {
    // toDetailString would probably be more useful to users, but lots of tests rely on the
    // current values.
    return getRootRelativePath().toString();
  }

  @SuppressWarnings("EqualsGetClass") // Distinct classes of Artifact are never equal.
  @Override
  public final boolean equals(Object other) {
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
    return equalsWithoutOwner(that) && ownersEqual(that);
  }

  final boolean equalsWithoutOwner(Artifact other) {
    return hashCode == other.hashCode && execPath.equals(other.execPath) && root.equals(other.root);
  }

  abstract boolean ownersEqual(Artifact other);

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

  /** Returns a string representing the complete artifact path information. */
  public final String toDetailString() {
    if (isSourceArtifact()) {
      // Source Artifact: relPath == execPath, & real path is not under execRoot
      return "[" + root + "]" + getRootRelativePathString();
    } else {
      // Derived Artifact: path and root are under execRoot
      //
      // TODO(blaze-team): this is misleading because execution_root isn't unique. Dig the
      // workspace name out and print that also.
      return "[[<execution_root>]" + root.getExecPath() + "]" + getRootRelativePathString();
    }
  }

  @Override
  public final SkyFunctionName functionName() {
    return ARTIFACT;
  }

  /** {@link Artifact#isSourceArtifact() is true.
   *
   * <p>Source artifacts have the property that unlike for output artifacts, direct file system
   * access for their contents should be safe, even in a distributed context.
   *
   * TODO(shahan): move {@link Artifact#getPath} to this subclass.
   */
  public static final class SourceArtifact extends Artifact {
    private final ArtifactOwner owner;

    @VisibleForTesting
    public SourceArtifact(ArtifactRoot root, PathFragment execPath, ArtifactOwner owner) {
      super(root, execPath);
      this.owner = owner;
    }

    /**
     * Source artifacts do not consider their owners in equality checks, since their owners are
     * purely cosmetic.
     */
    @Override
    boolean ownersEqual(Artifact other) {
      return true;
    }

    @Override
    public PathFragment getRootRelativePath() {
      return root.isExternalSourceRoot() ? getExecPath().subFragment(1) : getExecPath();
    }

    @Override
    public PathFragment getPathForLocationExpansion() {
      return getExecPath();
    }

    @Override
    public ArtifactOwner getArtifactOwner() {
      return owner;
    }

    @Override
    public Label getOwnerLabel() {
      return owner.getLabel();
    }

    boolean differentOwnerOrRoot(ArtifactOwner owner, ArtifactRoot root) {
      return !this.owner.equals(owner) || !this.getRoot().equals(root);
    }
  }

  /**
   * Special artifact types.
   *
   * @see SpecialArtifact
   */
  @VisibleForTesting
  public enum SpecialArtifactType {
    /** Google-specific legacy type. */
    FILESET,

    /**
     * A symlink. Not chased, can be dangling. All we care about is the return value of {@code
     * readlink()}.
     */
    UNRESOLVED_SYMLINK,

    /** A subtree containing multiple files and directories. */
    TREE,

    /** Special artifact type for workspace status information. */
    CONSTANT_METADATA,
  }

  /**
   * A special kind of artifact that either is a fileset or needs special metadata caching behavior.
   *
   * <p>We subclass {@link DerivedArtifact} instead of storing the special attributes inside in
   * order to save memory. The proportion of artifacts that are special is very small, and by not
   * having to keep around the attribute for the rest we save some memory.
   */
  @Immutable
  public static final class SpecialArtifact extends DerivedArtifact {
    private final SpecialArtifactType type;

    @VisibleForTesting
    public SpecialArtifact(
        ArtifactRoot root, PathFragment execPath, ActionLookupKey owner, SpecialArtifactType type) {
      super(root, execPath, owner, /*contentBasedPath=*/ false);
      this.type = type;
    }

    @Override
    public boolean isFileset() {
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
    public boolean isSymlink() {
      return type == SpecialArtifactType.UNRESOLVED_SYMLINK;
    }

    @Override
    public boolean hasParent() {
      return false;
    }

    @Override
    @Nullable
    public SpecialArtifact getParent() {
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

  // Keep in sync with DerivedArtifactCodec.
  @SuppressWarnings("unused") // Used by reflection
  private static class SpecialArtifactCodec implements ObjectCodec<SpecialArtifact> {
    @Override
    public Class<SpecialArtifact> getEncodedClass() {
      return SpecialArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, SpecialArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getGeneratingActionKey(), codedOut);
      context.serialize(obj.type, codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
    }

    @Override
    public SpecialArtifact deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ArtifactRoot root = context.deserialize(codedIn);
      ActionLookupData generatingActionKey = context.deserialize(codedIn);
      SpecialArtifactType type = context.deserialize(codedIn);
      SpecialArtifact artifact =
          new SpecialArtifact(
              root,
              DerivedArtifactCodec.validateAndGetRootExecPath(
                  root, generatingActionKey, context, codedIn),
              generatingActionKey.getActionLookupKey(),
              type);
      artifact.setGeneratingActionKey(generatingActionKey);
      return (SpecialArtifact)
          context.getDependency(ArtifactResolverSupplier.class).intern(artifact);
    }
  }

  /**
   * Artifact representing a single-file archive with the filesystem tree belonging to a {@linkplain
   * SpecialArtifact tree artifact}.
   *
   * <p>The archive is equivalent to the entire tree artifact -- it contains all of the {@linkplain
   * TreeFileArtifact children} (and nothing else) of the tree artifact with their filesystem
   * structure, relative to the {@linkplain SpecialArtifact#getExecPath() tree artifact directory}.
   */
  @AutoCodec
  public static final class ArchivedTreeArtifact extends DerivedArtifact {
    private static final PathFragment ARCHIVED_ARTIFACTS_DERIVED_TREE_ROOT =
        PathFragment.create(":archived_tree_artifacts");
    private final SpecialArtifact treeArtifact;

    public ArchivedTreeArtifact(
        SpecialArtifact treeArtifact, ArtifactRoot root, PathFragment execPath) {
      super(root, execPath, treeArtifact.getArtifactOwner());
      this.treeArtifact = treeArtifact;
    }

    @Override
    public SpecialArtifact getParent() {
      return treeArtifact;
    }

    /**
     * Creates an {@link ArchivedTreeArtifact} for a given tree artifact. Returned artifact is
     * stored in a permanent location, therefore can be shared across actions and builds.
     *
     * <p>Example: for a tree artifact of {@code bazel-out/k8-fastbuild/bin/directory} returns an
     * {@linkplain ArchivedTreeArtifact artifact} of: {@code
     * bazel-out/:archived_tree_artifacts/k8-fastbuild/bin/directory.zip}.
     */
    public static ArchivedTreeArtifact create(
        SpecialArtifact treeArtifact, PathFragment derivedPathPrefix) {
      return createWithCustomDerivedTreeRoot(
          treeArtifact,
          derivedPathPrefix,
          ARCHIVED_ARTIFACTS_DERIVED_TREE_ROOT,
          treeArtifact.getRootRelativePath().replaceName(treeArtifact.getFilename() + ".zip"));
    }

    /**
     * Creates an {@link ArchivedTreeArtifact} for a given tree artifact within provided derived
     * tree directory.
     *
     * <p>Example: for a tree artifact with root of {@code bazel-out/k8-fastbuild/bin} returns an
     * {@linkplain ArchivedTreeArtifact artifact} of: {@code
     * bazel-out/{customDerivedTreeRoot}/k8-fastbuild/bin/{rootRelativePath}} with root of: {@code
     * bazel-out/{customDerivedTreeRoot}/k8-fastbuild/bin}.
     */
    public static ArchivedTreeArtifact createWithCustomDerivedTreeRoot(
        SpecialArtifact treeArtifact,
        PathFragment derivedPathPrefix,
        PathFragment customDerivedTreeRoot,
        PathFragment rootRelativePath) {
      ArtifactRoot artifactRoot =
          createRootForArchivedArtifact(
              treeArtifact.getRoot(), derivedPathPrefix, customDerivedTreeRoot);
      ArchivedTreeArtifact archivedTreeArtifact =
          new ArchivedTreeArtifact(
              treeArtifact, artifactRoot, artifactRoot.getExecPath().getRelative(rootRelativePath));

      archivedTreeArtifact.setGeneratingActionKey(treeArtifact.getGeneratingActionKey());
      return archivedTreeArtifact;
    }

    private static ArtifactRoot createRootForArchivedArtifact(
        ArtifactRoot treeArtifactRoot,
        PathFragment derivedPathPrefix,
        PathFragment customDerivedTreeRoot) {
      Path execRoot = getExecRoot(treeArtifactRoot);

      // bazel-out/k8-fastbuild/bin -> bazel-out/{customDerivedTreeRoot}/k8-fastbuild/bin
      PathFragment rootExecPath =
          derivedPathPrefix
              .getRelative(customDerivedTreeRoot)
              .getRelative(treeArtifactRoot.getExecPath().relativeTo(derivedPathPrefix));

      return ArtifactRoot.asDerivedRoot(execRoot, rootExecPath);
    }

    private static Path getExecRoot(ArtifactRoot artifactRoot) {
      // /output_base/execroot/bazel-out/k8-fastbuild/bin
      Path rootPath = artifactRoot.getRoot().asPath();
      PathFragment rootPathFragment = rootPath.asFragment();
      // /output_base/execroot
      PathFragment execRootPath =
          rootPathFragment.subFragment(
              0, rootPathFragment.segmentCount() - artifactRoot.getExecPath().segmentCount());
      return rootPath.getFileSystem().getPath(execRootPath);
    }
  }

  /**
   * A special kind of artifact that represents a concrete file created at execution time under its
   * associated parent TreeArtifact.
   *
   * <p>TreeFileArtifacts should be only created during execution time inside some special actions
   * to support action inputs and outputs that are unpredictable at analysis time. TreeFileArtifacts
   * should not be created directly by any rules at analysis time.
   *
   * <p>There are two types of TreeFileArtifacts:
   *
   * <ol>
   *   <li>Outputs under a directory created by an action using {@code declare_directory}. In this
   *       case, a single action creates both the parent and all of the children. Instances should
   *       be created by calling {@link #createTreeOutput}. {@link #isChildOfDeclaredDirectory} will
   *       return {@code true}.
   *   <li>Outputs of an action template expansion. In this case, the parent directory is not
   *       actually produced by any action, but rather serves as a placeholder for dependant actions
   *       to declare a dep on during analysis, before the children are known. The children are
   *       created by various actions (from the template expansion). Instances should be created by
   *       calling {@link #createTemplateExpansionOutput}. {@link #isChildOfDeclaredDirectory} will
   *       return {@code false}.
   * </ol>
   */
  @Immutable
  @AutoCodec
  public static final class TreeFileArtifact extends DerivedArtifact {
    private final SpecialArtifact parent;
    private final PathFragment parentRelativePath;

    /**
     * Creates a {@link TreeFileArtifact} representing a child of the given parent tree artifact.
     *
     * <p>The child should already have been created by the parent's generating action. For this
     * reason, {@link DerivedArtifact#hasGeneratingActionKey} on the parent must be {@code true}
     * when this is called. The child is set with the same generating action.
     */
    public static TreeFileArtifact createTreeOutput(
        SpecialArtifact parent, PathFragment parentRelativePath) {
      Preconditions.checkArgument(
          parent.hasGeneratingActionKey(),
          "%s has no generating action key (parent owner: %s, parent relative path: %s)",
          parent,
          parent.getArtifactOwner(),
          parentRelativePath);
      ActionLookupData generatingActionKey = parent.getGeneratingActionKey();
      Preconditions.checkArgument(
          !isActionTemplateExpansionKey(generatingActionKey.getActionLookupKey()),
          "%s owned by action template expansion %s (parent relative path: %s)",
          parent,
          generatingActionKey.getActionLookupKey(),
          parentRelativePath);
      return createInternal(parent, parentRelativePath, generatingActionKey);
    }

    /**
     * Convenience method for {@link #createTreeOutput(SpecialArtifact, PathFragment)} with a string
     * relative path.
     */
    public static TreeFileArtifact createTreeOutput(
        SpecialArtifact parent, String parentRelativePath) {
      return createTreeOutput(parent, PathFragment.create(parentRelativePath));
    }

    /**
     * Creates a {@link TreeFileArtifact} representing the output of an action generated dynamically
     * by an {@link ActionTemplate} during the execution phase.
     *
     * <p>The returned artifact does not yet have a generating action set.
     */
    public static TreeFileArtifact createTemplateExpansionOutput(
        SpecialArtifact parent, PathFragment parentRelativePath, ActionLookupKey owner) {
      Preconditions.checkArgument(
          isActionTemplateExpansionKey(owner),
          "Template expansion outputs must be owned by an action template expansion key, but %s is"
              + " owned by %s (parent relative path: %s)",
          parent,
          owner,
          parentRelativePath);
      return new TreeFileArtifact(parent, parentRelativePath, owner);
    }

    /**
     * Convenience method for {@link #createTemplateExpansionOutput(SpecialArtifact, PathFragment,
     * ActionLookupKey)} with a string relative path.
     */
    public static TreeFileArtifact createTemplateExpansionOutput(
        SpecialArtifact parent, String parentRelativePath, ActionLookupKey owner) {
      return createTemplateExpansionOutput(parent, PathFragment.create(parentRelativePath), owner);
    }

    private TreeFileArtifact(
        SpecialArtifact parent, PathFragment parentRelativePath, ActionLookupKey owner) {
      super(
          parent.getRoot(),
          parent.getExecPath().getRelative(parentRelativePath),
          owner,
          /*contentBasedPath=*/ false);
      Preconditions.checkArgument(
          parent.isTreeArtifact(),
          "The parent of TreeFileArtifact (parent-relative path: %s) is not a TreeArtifact: %s",
          parentRelativePath,
          parent);
      Preconditions.checkArgument(
          !parentRelativePath.containsUplevelReferences() && !parentRelativePath.isAbsolute(),
          "%s is not a proper normalized relative path",
          parentRelativePath);
      this.parent = parent;
      this.parentRelativePath = parentRelativePath;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static TreeFileArtifact createInternal(
        SpecialArtifact parent,
        PathFragment parentRelativePath,
        ActionLookupData generatingActionKey) {
      TreeFileArtifact result =
          new TreeFileArtifact(
              parent, parentRelativePath, generatingActionKey.getActionLookupKey());
      result.setGeneratingActionKey(generatingActionKey);
      return result;
    }

    @Override
    public SpecialArtifact getParent() {
      return parent;
    }

    @Override
    public PathFragment getParentRelativePath() {
      return parentRelativePath;
    }

    @Override
    public String getTreeRelativePathString() {
      return parentRelativePath.getPathString();
    }

    @Override
    public boolean isChildOfDeclaredDirectory() {
      return !isActionTemplateExpansionKey(getArtifactOwner());
    }

    private static boolean isActionTemplateExpansionKey(ActionLookupKey key) {
      return SkyFunctions.ACTION_TEMPLATE_EXPANSION.equals(key.functionName());
    }
  }

  /** {@link ObjectCodec} for {@link SourceArtifact} */
  @SuppressWarnings("unused") // found by CLASSPATH-scanning magic
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

  public static final Function<Artifact, String> OUTPUT_DIR_RELATIVE_PATH_STRING =
      artifact -> artifact.getOutputDirRelativePath().getPathString();

  /**
   * Converts a collection of artifacts into execution-time path strings, and
   * adds those to a given collection. Middleman artifacts are ignored by this
   * method.
   */
  public static void addExecPaths(Iterable<Artifact> artifacts, Collection<String> output) {
    addNonMiddlemanArtifacts(artifacts, output, ActionInput::getExecPathString);
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
  public static Iterable<String> toRootRelativePaths(NestedSet<Artifact> artifacts) {
    return toRootRelativePaths(artifacts.toList());
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
    return Iterables.transform(
        Iterables.filter(artifacts, MIDDLEMAN_FILTER), ActionInput::getExecPathString);
  }

  /**
   * Lazily converts artifacts into output-dir-relative path strings. Middleman artifacts are
   * ignored by this method.
   */
  public static Iterable<String> toOutputDirRelativePaths(NestedSet<Artifact> artifacts) {
    return toOutputDirRelativePaths(artifacts.toList());
  }

  /**
   * Lazily converts artifacts into output-dir-relative path strings. Middleman artifacts are
   * ignored by this method.
   */
  public static Iterable<String> toOutputDirRelativePaths(Iterable<Artifact> artifacts) {
    return Iterables.transform(
        Iterables.filter(artifacts, MIDDLEMAN_FILTER),
        artifact -> artifact.getOutputDirRelativePath().getPathString());
  }

  /**
   * Converts a collection of artifacts into execution-time path strings, and returns those as an
   * immutable list. Middleman artifacts are ignored by this method.
   *
   * <p>Avoid this method in production code - it flattens the given nested set unconditionally.
   */
  @VisibleForTesting
  public static List<String> asExecPaths(NestedSet<Artifact> artifacts) {
    return asExecPaths(artifacts.toList());
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
   * Adds a collection of artifacts to a given collection, with {@link
   * MiddlemanType#AGGREGATING_MIDDLEMAN} middleman actions expanded once.
   */
  static void addExpandedArtifacts(
      Iterable<Artifact> artifacts,
      Collection<? super Artifact> output,
      ArtifactExpander artifactExpander) {
    addExpandedArtifacts(artifacts, output, Functions.<Artifact>identity(), artifactExpander);
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
  public void repr(Printer printer) {
    if (isSourceArtifact()) {
      printer.append("<source file " + getRootRelativePathString() + ">");
    } else {
      printer.append("<generated file " + getRootRelativePathString() + ">");
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

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("artifact", artifact)
          .add("owner", artifact.getArtifactOwner())
          .toString();
    }
  }
}
