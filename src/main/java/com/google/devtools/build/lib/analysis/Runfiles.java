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

package com.google.devtools.build.lib.analysis;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

import javax.annotation.Nullable;

/**
 * An object that encapsulates runfiles. Conceptually, the runfiles are a map of paths to files,
 * forming a symlink tree.
 *
 * <p>In order to reduce memory consumption, this map is not explicitly stored here, but instead as
 * a combination of four parts: artifacts placed at their root-relative paths, source tree symlinks,
 * root symlinks (outside of the source tree), and artifacts included as parts of "pruning
 * manifests" (see {@link PruningManifest}).
 */
@Immutable
@SkylarkModule(name = "runfiles", doc = "An interface for a set of runfiles.")
public final class Runfiles {
  private static final Function<SymlinkEntry, Artifact> TO_ARTIFACT =
      new Function<SymlinkEntry, Artifact>() {
        @Override
        public Artifact apply(SymlinkEntry input) {
          return input.getArtifact();
        }
      };

  private static final EmptyFilesSupplier DUMMY_EMPTY_FILES_SUPPLIER =
      new EmptyFilesSupplier() {
        @Override
        public Iterable<PathFragment> getExtraPaths(Set<PathFragment> manifestPaths) {
          return ImmutableList.of();
        }
      };

  private static final Function<Artifact, PathFragment> GET_ROOT_RELATIVE_PATH =
      new Function<Artifact, PathFragment>() {
        @Override
        public PathFragment apply(Artifact input) {
          return input.getRootRelativePath();
        }
      };

  private static final Function<PathFragment, String> PATH_FRAGMENT_TO_STRING =
      new Function<PathFragment, String>() {
        @Override
        public String apply(PathFragment input) {
          return input.toString();
        }
      };

  /**
   * An entry in the runfiles map.
   *
   * <p>build-runfiles.cc enforces the following constraints: The PathFragment must not be an
   * absolute path, nor contain "..".  Overlapping runfiles links are also refused. This is the case
   * where you ask to create a link to "foo" and also "foo/bar.txt". I.e. you're asking it to make
   * "foo" both a file (symlink) and a directory.
   *
   * <p>Links to directories are heavily discouraged.
   */
  //
  // O intrepid fixer or bugs and implementor of features, dare not to add a .equals() method
  // to this class, lest you condemn yourself, or a fellow other developer to spending two
  // delightful hours in a fancy hotel on a Chromebook that is utterly unsuitable for Java
  // development to figure out what went wrong, just like I just did.
  //
  // The semantics of the symlinks nested set dictates that later entries overwrite earlier
  // ones. However, the semantics of nested sets dictate that if there are duplicate entries, they
  // are only returned once in the iterator.
  //
  // These two things, innocent when taken alone, result in the effect that when there are three
  // entries for the same path, the first one and the last one the same, and the middle one
  // different, the *middle* one will take effect: the middle one overrides the first one, and the
  // first one prevents the last one from appearing on the iterator.
  //
  // The lack of a .equals() method prevents this by making the first entry in the above case not
  // equals to the third one if they are not the same instance (which they almost never are)
  //
  // Goodnight, prince(ss)?, and sweet dreams.
  private static final class SymlinkEntry {
    private final PathFragment path;
    private final Artifact artifact;

    private SymlinkEntry(PathFragment path, Artifact artifact) {
      this.path = path;
      this.artifact = artifact;
    }

    public PathFragment getPath() {
      return path;
    }

    public Artifact getArtifact() {
      return artifact;
    }

    @Override
    public String toString() {
      return path + " -> " + artifact.getRunfilesPath();
    }
  }

  // It is important to declare this *after* the DUMMY_SYMLINK_EXPANDER to avoid NPEs
  public static final Runfiles EMPTY = new Builder().build();

  /**
   * The directory to put all runfiles under.
   *
   * <p>Using "foo" will put runfiles under &lt;target&gt;.runfiles/foo.</p>
   *
   * <p>This is either set to the workspace name, or is empty.
   */
  private final PathFragment suffix;

  /**
   * The artifacts that should *always* be present in the runfiles directory. These are
   * differentiated from the artifacts that may or may not be included by a pruning manifest
   * (see {@link PruningManifest} below).
   *
   * <p>This collection may not include any middlemen. These artifacts will be placed at a location
   * that corresponds to the root-relative path of each artifact. It's possible for several
   * artifacts to have the same root-relative path, in which case the last one will win.
   */
  private final NestedSet<Artifact> unconditionalArtifacts;

  /**
   * A map of symlinks that should be present in the runfiles directory. In general, the symlink can
   * be determined from the artifact by using the root-relative path, so this should only be used
   * for cases where that isn't possible.
   *
   * <p>This may include runfiles symlinks from the root of the runfiles tree.
   */
  private final NestedSet<SymlinkEntry> symlinks;

  /**
   * A map of symlinks that should be present above the runfiles directory. These are useful for
   * certain rule types like AppEngine apps which have root level config files outside of the
   * regular source tree.
   */
  private final NestedSet<SymlinkEntry> rootSymlinks;

  /**
   * Interface used for adding empty files to the runfiles at the last minute. Mainly to support
   * python-related rules adding __init__.py files.
   */
  public interface EmptyFilesSupplier {
    /** Calculate additional empty files to add based on the existing manifest paths. */
    Iterable<PathFragment> getExtraPaths(Set<PathFragment> manifestPaths);
  }

  /** Generates extra (empty file) inputs. */
  private final EmptyFilesSupplier emptyFilesSupplier;

  /**
   * Behavior upon finding a conflict between two runfile entries. A conflict means that two
   * different artifacts have the same runfiles path specified.  For example, adding artifact
   * "a.foo" at path "bar" when there is already an artifact "b.foo" at path "bar".  The policies
   * are ordered from least strict to most strict.
   *
   * <p>Note that conflicts are found relatively late, when the manifest file is created, not when
   * the symlinks are added to runfiles.
   *
   * <p>If no EventHandler is available, all values are treated as IGNORE.
   */
  public enum ConflictPolicy {
    IGNORE,
    WARN,
    ERROR,
  }

  /** Policy for this Runfiles tree */
  private ConflictPolicy conflictPolicy = ConflictPolicy.IGNORE;

  /**
   * Defines a set of artifacts that may or may not be included in the runfiles directory and
   * a manifest file that makes that determination. These are applied on top of any artifacts
   * specified in {@link #unconditionalArtifacts}.
   *
   * <p>The incentive behind this is to enable execution-phase "pruning" of runfiles. Anything
   * set in unconditionalArtifacts is hard-set in Blaze's analysis phase, and thus unchangeable in
   * response to execution phase results. This isn't always convenient. For example, say we have an
   * action that consumes a set of "possible" runtime dependencies for a source file, parses that
   * file for "import a.b.c" statements, and outputs a manifest of the actual dependencies that are
   * referenced and thus really needed. This can reduce the size of the runfiles set, but we can't
   * use this information until the manifest output is available.
   *
   * <p>Only artifacts present in the candidate set AND the manifest output make it into the
   * runfiles tree. The candidate set requirement guarantees that analysis-time dependencies are a
   * superset of the pruned dependencies, so undeclared inclusions (which can break build
   * correctness) aren't possible.
   */
  public static class PruningManifest {
    private final NestedSet<Artifact> candidateRunfiles;
    private final Artifact manifestFile;

    /**
     * Creates a new pruning manifest.
     *
     * @param candidateRunfiles set of possible artifacts that the manifest file may reference
     * @param manifestFile the manifest file, expected to be a newline-separated list of
     *     source tree root-relative paths (i.e. "my/package/myfile.txt"). Anything that can't be
     *     resolved back to an entry in candidateRunfiles is ignored and will *not* make it into
     *     the runfiles tree.
     */
    public PruningManifest(NestedSet<Artifact> candidateRunfiles, Artifact manifestFile) {
      this.candidateRunfiles = candidateRunfiles;
      this.manifestFile = manifestFile;
    }

    public NestedSet<Artifact> getCandidateRunfiles() {
      return candidateRunfiles;
    }

    public Artifact getManifestFile() {
      return manifestFile;
    }
  }

  /**
   * The pruning manifests that should be applied to these runfiles.
   */
  private final NestedSet<PruningManifest> pruningManifests;

  private final boolean legacyRepositoryStructure;

  private Runfiles(PathFragment suffix,
      NestedSet<Artifact> artifacts,
      NestedSet<SymlinkEntry> symlinks,
      NestedSet<SymlinkEntry> rootSymlinks,
      NestedSet<PruningManifest> pruningManifests,
      EmptyFilesSupplier emptyFilesSupplier,
      ConflictPolicy conflictPolicy,
      boolean legacyRepositoryStructure) {
    this.suffix = suffix;
    this.unconditionalArtifacts = Preconditions.checkNotNull(artifacts);
    this.symlinks = Preconditions.checkNotNull(symlinks);
    this.rootSymlinks = Preconditions.checkNotNull(rootSymlinks);
    this.pruningManifests = Preconditions.checkNotNull(pruningManifests);
    this.emptyFilesSupplier = Preconditions.checkNotNull(emptyFilesSupplier);
    this.conflictPolicy = conflictPolicy;
    this.legacyRepositoryStructure = legacyRepositoryStructure;
  }

  /**
   * Returns the runfiles' suffix.
   */
  public PathFragment getSuffix() {
    return suffix;
  }

  /**
   * Returns the artifacts that are unconditionally included in the runfiles (as opposed to
   * pruning manifest candidates, which may or may not be included).
   */
  public NestedSet<Artifact> getUnconditionalArtifacts() {
    return unconditionalArtifacts;
  }

  /**
   * Returns the artifacts that are unconditionally included in the runfiles (as opposed to
   * pruning manifest candidates, which may or may not be included). Middleman artifacts are
   * excluded.
   */
  public Iterable<Artifact> getUnconditionalArtifactsWithoutMiddlemen() {
    return Iterables.filter(unconditionalArtifacts, Artifact.MIDDLEMAN_FILTER);
  }

  /**
   * Returns the collection of runfiles as artifacts, including both unconditional artifacts
   * and pruning manifest candidates.
   */
  @SkylarkCallable(
    name = "files",
    doc = "Returns the set of runfiles as files",
    structField = true
  )
  public NestedSet<Artifact> getArtifacts() {
    NestedSetBuilder<Artifact> allArtifacts = NestedSetBuilder.stableOrder();
    allArtifacts.addAll(unconditionalArtifacts.toCollection());
    for (PruningManifest manifest : getPruningManifests()) {
      allArtifacts.addTransitive(manifest.getCandidateRunfiles());
    }
    return allArtifacts.build();
  }

  /**
   * Returns the collection of runfiles as artifacts, including both unconditional artifacts
   * and pruning manifest candidates. Middleman artifacts are excluded.
   */
  public Iterable<Artifact> getArtifactsWithoutMiddlemen() {
    return Iterables.filter(getArtifacts(), Artifact.MIDDLEMAN_FILTER);
  }

  /**
   * Returns the symlinks.
   */
  @SkylarkCallable(name = "symlinks", doc = "Returns the set of symlinks", structField = true)
  public NestedSet<SymlinkEntry> getSymlinks() {
    return symlinks;
  }

  @SkylarkCallable(
    name = "empty_filenames",
    doc = "Returns names of empty files to create.",
    structField = true
  )
  public NestedSet<String> getEmptyFilenames() {
    Set<PathFragment> manifest = new TreeSet<>();
    Iterables.addAll(
        manifest, Iterables.transform(getArtifacts().toCollection(), GET_ROOT_RELATIVE_PATH));
    return NestedSetBuilder.wrap(
        Order.STABLE_ORDER,
        Iterables.transform(emptyFilesSupplier.getExtraPaths(manifest), PATH_FRAGMENT_TO_STRING));
  }

  /**
   * Returns the symlinks as a map from path fragment to artifact.
   */
  public Map<PathFragment, Artifact> getSymlinksAsMap() {
    return new ManifestBuilder(ConflictChecker.IGNORE_CHECKER, suffix, legacyRepositoryStructure)
        .putSymlinks(symlinks).build();
  }

  /**
   * Returns the symlinks as a map from PathFragment to Artifact.
   *
   * @param eventHandler Used for throwing an error if we have an obscuring runlink within the
   *    normal source tree entries, or runfile conflicts. May be null, in which case obscuring
   *    symlinks are silently discarded, and conflicts are overwritten.
   * @param location Location for eventHandler warnings. Ignored if eventHandler is null.
   * @return Map<PathFragment, Artifact> path fragment to artifact, of normal source tree entries
   *    and elements that live outside the source tree. Null values represent empty input files.
   */
  public Map<PathFragment, Artifact> getRunfilesInputs(EventHandler eventHandler,
      Location location) throws IOException {
    ManifestBuilder builder = new ManifestBuilder(
        new ConflictChecker(conflictPolicy, eventHandler, location),
        suffix,
        legacyRepositoryStructure);
    builder.putSymlinks(symlinks);
    // Add unconditional artifacts (committed to inclusion on construction of runfiles).
    for (Artifact artifact : getUnconditionalArtifactsWithoutMiddlemen()) {
      builder.put(RunfilesPath.resolve(artifact.getRootRelativePath(), suffix), artifact);
    }

    // Add conditional artifacts (only included if they appear in a pruning manifest).
    for (Runfiles.PruningManifest pruningManifest : getPruningManifests()) {
      // This map helps us convert from source tree root-relative paths back to artifacts.
      Map<String, Artifact> allowedRunfiles = new HashMap<>();
      for (Artifact artifact : pruningManifest.getCandidateRunfiles()) {
        allowedRunfiles.put(artifact.getRootRelativePath().getPathString(), artifact);
      }
      try (BufferedReader reader = new BufferedReader(
          new InputStreamReader(pruningManifest.getManifestFile().getPath().getInputStream()))) {
        String line;
        while ((line = reader.readLine()) != null) {
          Artifact artifact = allowedRunfiles.get(line);
          if (artifact != null) {
            builder.put(RunfilesPath.resolve(artifact.getRootRelativePath(), suffix), artifact);
          }
        }
      }
    }
    builder.filterListForObscuringSymlinks(eventHandler, location);

    // TODO(bazel-team): Create /dev/null-like Artifact to avoid nulls?
    for (PathFragment extraPath : emptyFilesSupplier.getExtraPaths(builder.getPaths())) {
      builder.put(RunfilesPath.alreadyResolved(extraPath, suffix), null);
    }

    // Finally add symlinks relative to the root of the runfiles tree, on top of everything else.
    // This operation is always checked for conflicts, to match historical behavior.
    if (conflictPolicy == ConflictPolicy.IGNORE) {
      builder.resetConflictPolicy(
          new ConflictChecker(ConflictPolicy.WARN, eventHandler, location));
    }
    builder.putRootSymlinks(rootSymlinks);
    return builder.build();
  }

  /**
   * Returns the root symlinks.
   */
  public NestedSet<SymlinkEntry> getRootSymlinks() {
    return rootSymlinks;
  }

  /**
   * Returns the root symlinks as a map from path fragment to artifact.
   *
   * @param checker If not null, check for conflicts using this checker.
   */
  public Map<PathFragment, Artifact> getRootSymlinksAsMap(@Nullable ConflictChecker checker) {
    return new ManifestBuilder(checker, suffix, legacyRepositoryStructure)
        .putRootSymlinks(rootSymlinks)
        .buildWithoutDummyFile();
  }

  /**
   * Returns the unified map of path fragments to artifacts, taking both artifacts and symlinks into
   * account.
   */
  public Map<PathFragment, Artifact> asMapWithoutRootSymlinks() {
    ManifestBuilder builder = new ManifestBuilder(
        ConflictChecker.IGNORE_CHECKER, suffix, legacyRepositoryStructure).putSymlinks(symlinks);
    // If multiple artifacts have the same root-relative path, the last one in the list will win.
    // That is because the runfiles tree cannot contain the same artifact for different
    // configurations, because it only uses root-relative paths.
    for (Artifact artifact : Iterables.filter(unconditionalArtifacts, Artifact.MIDDLEMAN_FILTER)) {
      builder.put(RunfilesPath.resolve(artifact.getRootRelativePath(), suffix), artifact);
    }
    return builder.build();
  }

  /**
   * Returns the pruning manifests specified for this runfiles tree.
   */
  public NestedSet<PruningManifest> getPruningManifests() {
    return pruningManifests;
  }

  /**
   * Returns the manifest expander specified for this runfiles tree.
   */
  private EmptyFilesSupplier getEmptyFilesProvider() {
    return emptyFilesSupplier;
  }

  /**
   * Returns the unified map of path fragments to artifacts, taking into account artifacts,
   * symlinks, and pruning manifest candidates. The returned set is guaranteed to be a (not
   * necessarily strict) superset of the actual runfiles tree created at execution time.
   */
  public NestedSet<Artifact> getAllArtifacts() {
    if (isEmpty()) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    NestedSetBuilder<Artifact> allArtifacts = NestedSetBuilder.stableOrder();
    allArtifacts
        .addTransitive(unconditionalArtifacts)
        .addAll(Iterables.transform(symlinks, TO_ARTIFACT))
        .addAll(Iterables.transform(rootSymlinks, TO_ARTIFACT));
    for (PruningManifest manifest : getPruningManifests()) {
      allArtifacts.addTransitive(manifest.getCandidateRunfiles());
    }
    return allArtifacts.build();
  }

  /**
   * Returns if there are no runfiles.
   */
  public boolean isEmpty() {
    return unconditionalArtifacts.isEmpty() && symlinks.isEmpty() && rootSymlinks.isEmpty() &&
        pruningManifests.isEmpty();
  }

  /** Returns currently policy for conflicting symlink entries. */
  public ConflictPolicy getConflictPolicy() {
    return this.conflictPolicy;
  }

  /** Set whether we should warn about conflicting symlink entries. */
  public Runfiles setConflictPolicy(ConflictPolicy conflictPolicy) {
    this.conflictPolicy = conflictPolicy;
    return this;
  }

  /**
   * Helper class to make sure that every path added to runfiles is relative to the root of the
   * runfiles tree.
   */
  @VisibleForTesting
  static class RunfilesPath {

    private final PathFragment path;
    private final boolean external;

    public static RunfilesPath resolve(PathFragment path, PathFragment workspaceName) {
      return new RunfilesPath(makeRelativeToRunfilesDir(path, workspaceName), workspaceName);
    }

    public static RunfilesPath alreadyResolved(PathFragment path, PathFragment workspaceName) {
      return new RunfilesPath(path, workspaceName);
    }

    private RunfilesPath(PathFragment path, PathFragment workspaceName) {
      this.path = path;
      this.external = path.segmentCount() > 1 && !path.startsWith(workspaceName);
    }

    public PathFragment getPath() {
      return path;
    }

    /**
     * Returns if this file is from an external repository.
     */
    public boolean isExternal() {
      return external;
    }

    /**
     * This takes an execution-root-relative path and turns it into a runfiles-relative path. For
     * paths in the current repository, it prefixes them with the workspace name. For paths in
     * external repositories, it turns the execution root path (external/repo-name/foo) into a
     * runfiles path (repo-name/foo).
     */
    private static PathFragment makeRelativeToRunfilesDir(
        PathFragment path, PathFragment mainWorkspace) {
      if (path.getSegment(0).equals(Label.EXTERNAL_PATH_PREFIX)) {
        path = path.relativeTo(Label.EXTERNAL_PACKAGE_NAME);
      } else {
        path = mainWorkspace.getRelative(path);
      }
      return path;
    }
  }

  /**
   * A builder to handle the logic of creating a manifest mapping.
   */
  @VisibleForTesting
  static final class ManifestBuilder {
    private final PathFragment workspaceName;
    private final boolean legacyRunfilesStructure;

    private Map<PathFragment, Artifact> map;
    private ConflictChecker checker;
    private boolean sawWorkspaceName;

    ManifestBuilder(
        ConflictChecker checker, PathFragment workspaceName, boolean legacyRepositoryStructure) {
      this.workspaceName = workspaceName;
      this.legacyRunfilesStructure = legacyRepositoryStructure;
      this.map = new LinkedHashMap<>();
      this.checker = checker == null ? ConflictChecker.IGNORE_CHECKER : checker;
      this.sawWorkspaceName = false;
    }

    public void resetConflictPolicy(ConflictChecker checker) {
      this.checker = checker;
    }

    public ManifestBuilder put(RunfilesPath runfilesPath, Artifact artifact) {
      checker.check(map, runfilesPath, artifact);
      PathFragment path = runfilesPath.getPath();
      if (path.startsWith(workspaceName)) {
        sawWorkspaceName = true;
      }
      if (runfilesPath.isExternal() && legacyRunfilesStructure) {
        // Store runfiles at both .runfiles/wsname/external/foo and .runfiles/foo, to allow people
        // time to migrate to the second form.
        map.put(
            workspaceName.getRelative(Label.EXTERNAL_PACKAGE_NAME).getRelative(path), artifact);
      }
      map.put(path, artifact);
      return this;
    }

    /**
     * Flatten a sequence of entries into a single map.
     *
     * @param symlinks Sequence of entries to add.
     */
    public ManifestBuilder putSymlinks(NestedSet<SymlinkEntry> symlinks) {
      for (SymlinkEntry entry : symlinks) {
        put(RunfilesPath.resolve(entry.getPath(), workspaceName), entry.getArtifact());
      }
      return this;
    }

    /**
     * Flatten a sequence of entries into a single map. Symlink entries are relative to .runfiles,
     * not .runfiles/wsname, so it's assumed that external workspace entries are already resolved.
     *
     * @param symlinks Sequence of entries to add.
     */
    public ManifestBuilder putRootSymlinks(NestedSet<SymlinkEntry> symlinks) {
      for (SymlinkEntry entry : symlinks) {
        put(RunfilesPath.alreadyResolved(entry.getPath(), workspaceName), entry.getArtifact());
      }
      return this;
    }

    /**
     * This destroys the existing listing and replaces it with one that has no obscuring symlinks.
     *
     * @param eventHandler Used for throwing an error if we have an obscuring runlink.
     *                 May be null, in which case obscuring symlinks are silently discarded.
     * @param location Location for reporter. Ignored if reporter is null.
     */
    @VisibleForTesting
    ManifestBuilder filterListForObscuringSymlinks(EventHandler eventHandler, Location location) {
      Map<PathFragment, Artifact> newManifest = new HashMap<>();

      outer:
      for (Entry<PathFragment, Artifact> entry : map.entrySet()) {
        PathFragment source = entry.getKey();
        Artifact symlink = entry.getValue();
        // drop nested entries; warn if this changes anything
        int n = source.segmentCount();
        for (int j = 1; j < n; ++j) {
          PathFragment prefix = source.subFragment(0, n - j);
          Artifact ancestor = map.get(prefix);
          if (ancestor != null) {
            // This is an obscuring symlink, so just drop it and move on if there's no reporter.
            if (eventHandler == null) {
              continue outer;
            }
            PathFragment suffixPath = source.subFragment(n - j, n);
            Path viaAncestor = ancestor.getPath().getRelative(suffixPath);
            Path expected = symlink.getPath();
            if (!viaAncestor.equals(expected)) {
              eventHandler.handle(Event.warn(location, "runfiles symlink " + source + " -> "
                  + expected + " obscured by " + prefix + " -> " + ancestor.getPath()));
            }
            continue outer;
          }
        }
        newManifest.put(entry.getKey(), entry.getValue());
      }

      map = newManifest;
      return this;
    }

    public Set<PathFragment> getPaths() {
      return map.keySet();
    }

    /**
     * Returns the map without checking if the main repository's directory needs to be added to
     * the runfiles tree.
     */
    public Map<PathFragment, Artifact> buildWithoutDummyFile() {
      return map;
    }

    public Map<PathFragment, Artifact> build() {
      if (!sawWorkspaceName && !map.isEmpty()) {
        // If we haven't seen it and we have seen other files, add the workspace name directory.
        // It might not be there if all of the runfiles are from other repos (and then running from
        // x.runfiles/ws will fail, because ws won't exist). We can't tell Runfiles to create a
        // directory, so instead this creates a hidden file inside the desired directory.
        map.put(workspaceName.getRelative(".runfile"), null);
      }
      return map;
    }
  }

  /**
   * Checks for conflicts between entries in a runfiles tree while putting them in a map.
   */
  public static final class ConflictChecker {
    /** Prebuilt ConflictChecker with policy set to IGNORE */
    public static final ConflictChecker IGNORE_CHECKER =
        new ConflictChecker(ConflictPolicy.IGNORE, null, null);

    /** Behavior when a conflict is found. */
    private final ConflictPolicy policy;

    /** Used for warning on conflicts. May be null, in which case conflicts are ignored. */
    private final EventHandler eventHandler;

    /** Location for eventHandler warnings. Ignored if eventHandler is null. */
    private final Location location;

    /** Type of event to emit */
    private final EventKind eventKind;

    /** Construct a ConflictChecker for the given reporter with the given behavior */
    public ConflictChecker(ConflictPolicy policy, EventHandler eventHandler, Location location) {
      if (eventHandler == null) {
        this.policy = ConflictPolicy.IGNORE; // Can't warn even if we wanted to
      } else {
        this.policy = policy;
      }
      this.eventHandler = eventHandler;
      this.location = location;
      this.eventKind = (policy == ConflictPolicy.ERROR) ? EventKind.ERROR : EventKind.WARNING;
    }

    /**
     * Add an entry to a Map of symlinks, optionally reporting conflicts.
     *
     * @param runfilesPath Path relative to the .runfiles directory, used as key in map.
     * @param artifact Artifact to store in map. This may be null to indicate an empty file.
     */
    void check(Map<PathFragment, Artifact> map, RunfilesPath runfilesPath, Artifact artifact) {
      PathFragment path = runfilesPath.getPath();
      if (policy != ConflictPolicy.IGNORE && map.containsKey(path)) {
        // Previous and new entry might have value of null
        Artifact previous = map.get(path);
        if (!Objects.equals(previous, artifact)) {
          String previousStr = (previous == null) ? "empty file" : previous.getPath().toString();
          String artifactStr = (artifact == null) ? "empty file" : artifact.getPath().toString();
          String message =
              String.format(
                  "overwrote runfile %s, was symlink to %s, now symlink to %s",
                  path.getSafePathString(),
                  previousStr,
                  artifactStr);
          eventHandler.handle(Event.of(eventKind, location, message));
        }
      }
    }
  }

  /**
   * Builder for Runfiles objects.
   */
  public static final class Builder {

    /** This is set to the workspace name */
    private PathFragment suffix;
    /**
     * If external runfiles should be under .runfiles/wsname/external/repo (in addition to
     * .runfiles/repo).
     * TODO(kchodorow): remove this once the old form is deprecated.
     */
    private final boolean legacyRepositoryStructure;

    /**
     * This must be COMPILE_ORDER because {@link #asMapWithoutRootSymlinks} overwrites earlier
     * entries with later ones, so we want a post-order iteration.
     */
    private NestedSetBuilder<Artifact> artifactsBuilder =
        NestedSetBuilder.compileOrder();
    private NestedSetBuilder<SymlinkEntry> symlinksBuilder =
        NestedSetBuilder.stableOrder();
    private NestedSetBuilder<SymlinkEntry> rootSymlinksBuilder =
        NestedSetBuilder.stableOrder();
    private NestedSetBuilder<PruningManifest> pruningManifestsBuilder =
        NestedSetBuilder.stableOrder();
    private EmptyFilesSupplier emptyFilesSupplier = DUMMY_EMPTY_FILES_SUPPLIER;

    /** Build the Runfiles object with this policy */
    private ConflictPolicy conflictPolicy = ConflictPolicy.IGNORE;

    /**
     * Only used for Runfiles.EMPTY.
     */
    private Builder() {
      this.suffix = PathFragment.EMPTY_FRAGMENT;
      this.legacyRepositoryStructure = false;
    }

    /**
     * Creates a builder with the given suffix. Transitional constructor so that new rules don't
     * accidentally depend on the legacy repository structure, until that option is removed.
     *
     * @param workspace is the string specified in workspace() in the WORKSPACE file.
     */
    public Builder(String workspace) {
      this(workspace, false);
    }

    /**
     * Creates a builder with the given suffix.
     * @param workspace is the string specified in workspace() in the WORKSPACE file.
     * @param legacyRepositoryStructure if the wsname/external/repo symlinks should also be
     *     created.
     */
    public Builder(String workspace, boolean legacyRepositoryStructure) {
      this.suffix = new PathFragment(workspace);
      this.legacyRepositoryStructure = legacyRepositoryStructure;
    }

    /**
     * Builds a new Runfiles object.
     */
    public Runfiles build() {
      return new Runfiles(suffix, artifactsBuilder.build(), symlinksBuilder.build(),
          rootSymlinksBuilder.build(), pruningManifestsBuilder.build(),
          emptyFilesSupplier, conflictPolicy, legacyRepositoryStructure);
    }

    /**
     * Adds an artifact to the internal collection of artifacts.
     */
    public Builder addArtifact(Artifact artifact) {
      Preconditions.checkNotNull(artifact);
      artifactsBuilder.add(artifact);
      return this;
    }

    /**
     * Adds several artifacts to the internal collection.
     */
    public Builder addArtifacts(Iterable<Artifact> artifacts) {
      for (Artifact artifact : artifacts) {
        addArtifact(artifact);
      }
      return this;
    }


    /**
     * @deprecated Use {@link #addTransitiveArtifacts} instead, to prevent increased memory use.
     */
    @Deprecated
    public Builder addArtifacts(NestedSet<Artifact> artifacts) {
      // Do not delete this method, or else addArtifacts(Iterable) calls with a NestedSet argument
      // will not be flagged.
      Iterable<Artifact> it = artifacts;
      addArtifacts(it);
      return this;
    }

    /**
     * Adds a nested set to the internal collection.
     */
    public Builder addTransitiveArtifacts(NestedSet<Artifact> artifacts) {
      artifactsBuilder.addTransitive(artifacts);
      return this;
    }

    /**
     * Adds a symlink.
     */
    public Builder addSymlink(PathFragment link, Artifact target) {
      Preconditions.checkNotNull(link);
      Preconditions.checkNotNull(target);
      symlinksBuilder.add(new SymlinkEntry(link, target));
      return this;
    }

    /**
     * Adds several symlinks.
     */
    public Builder addSymlinks(Map<PathFragment, Artifact> symlinks) {
      for (Map.Entry<PathFragment, Artifact> symlink : symlinks.entrySet()) {
        symlinksBuilder.add(new SymlinkEntry(symlink.getKey(), symlink.getValue()));
      }
      return this;
    }

    /**
     * Adds several symlinks as a NestedSet.
     */
    public Builder addSymlinks(NestedSet<SymlinkEntry> symlinks) {
      symlinksBuilder.addTransitive(symlinks);
      return this;
    }

    /**
     * Adds a root symlink.
     */
    public Builder addRootSymlink(PathFragment link, Artifact target) {
      Preconditions.checkNotNull(link);
      Preconditions.checkNotNull(target);
      rootSymlinksBuilder.add(new SymlinkEntry(link, target));
      return this;
    }

    /**
     * Adds several root symlinks.
     */
    public Builder addRootSymlinks(Map<PathFragment, Artifact> symlinks) {
      for (Map.Entry<PathFragment, Artifact> symlink : symlinks.entrySet()) {
        rootSymlinksBuilder.add(new SymlinkEntry(symlink.getKey(), symlink.getValue()));
      }
      return this;
    }

    /**
     * Adds several root symlinks as a NestedSet.
     */
    public Builder addRootSymlinks(NestedSet<SymlinkEntry> symlinks) {
      rootSymlinksBuilder.addTransitive(symlinks);
      return this;
    }

    /**
     * Adds a pruning manifest. See {@link PruningManifest} for an explanation.
     */
    public Builder addPruningManifest(PruningManifest manifest) {
      pruningManifestsBuilder.add(manifest);
      return this;
    }

    /**
     * Adds several pruning manifests as a NestedSet. See {@link PruningManifest} for an
     * explanation.
     */
    public Builder addPruningManifests(NestedSet<PruningManifest> manifests) {
      pruningManifestsBuilder.addTransitive(manifests);
      return this;
    }

    /**
     * Specify a function that can create additional manifest entries based on the input entries,
     * see {@link EmptyFilesSupplier} for more details.
     */
    public Builder setEmptyFilesSupplier(EmptyFilesSupplier supplier) {
      emptyFilesSupplier = Preconditions.checkNotNull(supplier);
      return this;
    }

    /**
     * Merges runfiles from a given runfiles support.
     *
     * @param runfilesSupport the runfiles support to be merged in
     */
    public Builder merge(@Nullable RunfilesSupport runfilesSupport) {
      if (runfilesSupport == null) {
        return this;
      }
      // TODO(bazel-team): We may be able to remove this now.
      addArtifact(runfilesSupport.getRunfilesMiddleman());
      merge(runfilesSupport.getRunfiles());
      return this;
    }

    /**
     * Adds the runfiles for a particular target and visits the transitive closure of "srcs",
     * "deps" and "data", collecting all of their respective runfiles.
     */
    public Builder addRunfiles(RuleContext ruleContext,
        Function<TransitiveInfoCollection, Runfiles> mapping) {
      Preconditions.checkNotNull(mapping);
      Preconditions.checkNotNull(ruleContext);
      addDataDeps(ruleContext);
      addNonDataDeps(ruleContext, mapping);
      return this;
    }

    /**
     * Adds the files specified by a mapping from the transitive info collection to the runfiles.
     *
     * <p>Dependencies in {@code srcs} and {@code deps} are considered.
     */
    public Builder add(RuleContext ruleContext,
        Function<TransitiveInfoCollection, Runfiles> mapping) {
      Preconditions.checkNotNull(ruleContext);
      Preconditions.checkNotNull(mapping);
      for (TransitiveInfoCollection dep : getNonDataDeps(ruleContext)) {
        Runfiles runfiles = mapping.apply(dep);
        if (runfiles != null) {
          merge(runfiles);
        }
      }

      return this;
    }

    /**
     * Collects runfiles from data dependencies of a target.
     */
    public Builder addDataDeps(RuleContext ruleContext) {
      addTargets(getPrerequisites(ruleContext, "data", Mode.DATA), RunfilesProvider.DATA_RUNFILES);
      return this;
    }

    /**
     * Collects runfiles from "srcs" and "deps" of a target.
     */
    public Builder addNonDataDeps(RuleContext ruleContext,
        Function<TransitiveInfoCollection, Runfiles> mapping) {
      for (TransitiveInfoCollection target : getNonDataDeps(ruleContext)) {
        addTargetExceptFileTargets(target, mapping);
      }
      return this;
    }

    public Builder addTargets(Iterable<? extends TransitiveInfoCollection> targets,
        Function<TransitiveInfoCollection, Runfiles> mapping) {
      for (TransitiveInfoCollection target : targets) {
        addTarget(target, mapping);
      }
      return this;
    }

    public Builder addTarget(TransitiveInfoCollection target,
        Function<TransitiveInfoCollection, Runfiles> mapping) {
      return addTargetIncludingFileTargets(target, mapping);
    }

    private Builder addTargetExceptFileTargets(TransitiveInfoCollection target,
        Function<TransitiveInfoCollection, Runfiles> mapping) {
      Runfiles runfiles = mapping.apply(target);
      if (runfiles != null) {
        merge(runfiles);
      }

      return this;
    }

    private Builder addTargetIncludingFileTargets(TransitiveInfoCollection target,
        Function<TransitiveInfoCollection, Runfiles> mapping) {
      if (target.getProvider(RunfilesProvider.class) == null
          && mapping == RunfilesProvider.DATA_RUNFILES) {
        // RuleConfiguredTarget implements RunfilesProvider, so this will only be called on
        // FileConfiguredTarget instances.
        // TODO(bazel-team): This is a terrible hack. We should be able to make this go away
        // by implementing RunfilesProvider on FileConfiguredTarget. We'd need to be mindful
        // of the memory use, though, since we have a whole lot of FileConfiguredTarget instances.
        addTransitiveArtifacts(target.getProvider(FileProvider.class).getFilesToBuild());
        return this;
      }

      return addTargetExceptFileTargets(target, mapping);
    }

    /**
     * Adds symlinks to given artifacts at their exec paths.
     */
    public Builder addSymlinksToArtifacts(Iterable<Artifact> artifacts) {
      for (Artifact artifact : artifacts) {
        addSymlink(artifact.getExecPath(), artifact);
      }
      return this;
    }

    /**
     * Add the other {@link Runfiles} object transitively.
     */
    public Builder merge(Runfiles runfiles) {
      return merge(runfiles, true);
    }

    /**
     * Add the other {@link Runfiles} object transitively, but don't merge
     * pruning manifests.
     */
    public Builder mergeExceptPruningManifests(Runfiles runfiles) {
      return merge(runfiles, false);
    }

    /**
     * Add the other {@link Runfiles} object transitively, with the option to include or exclude
     * pruning manifests in the merge.
     */
    private Builder merge(Runfiles runfiles, boolean includePruningManifests) {
      // Propagate the most strict conflict checking from merged-in runfiles
      if (runfiles.conflictPolicy.compareTo(conflictPolicy) > 0) {
        conflictPolicy = runfiles.conflictPolicy;
      }
      if (runfiles.isEmpty()) {
        return this;
      }
      // The suffix should be the same within any blaze build, except for the EMPTY runfiles, which
      // may have an empty suffix, but that is covered above.
      Preconditions.checkArgument(suffix.equals(runfiles.suffix));
      artifactsBuilder.addTransitive(runfiles.getUnconditionalArtifacts());
      symlinksBuilder.addTransitive(runfiles.getSymlinks());
      rootSymlinksBuilder.addTransitive(runfiles.getRootSymlinks());
      if (includePruningManifests) {
        pruningManifestsBuilder.addTransitive(runfiles.getPruningManifests());
      }
      if (emptyFilesSupplier == DUMMY_EMPTY_FILES_SUPPLIER) {
        emptyFilesSupplier = runfiles.getEmptyFilesProvider();
      } else {
        EmptyFilesSupplier otherSupplier = runfiles.getEmptyFilesProvider();
        Preconditions.checkState((otherSupplier == DUMMY_EMPTY_FILES_SUPPLIER)
          || emptyFilesSupplier.equals(otherSupplier));
      }
      return this;
    }

    private static Iterable<TransitiveInfoCollection> getNonDataDeps(RuleContext ruleContext) {
      return Iterables.concat(
          // TODO(bazel-team): This line shouldn't be here. Removing it requires that no rules have
          // dependent rules in srcs (except for filegroups and such), but always in deps.
          // TODO(bazel-team): DONT_CHECK is not optimal here. Rules that use split configs need to
          // be changed not to call into here.
          getPrerequisites(ruleContext, "srcs", Mode.DONT_CHECK),
          getPrerequisites(ruleContext, "deps", Mode.DONT_CHECK));
    }

    /**
     * For the specified attribute "attributeName" (which must be of type list(label)), resolves all
     * the labels into ConfiguredTargets (for the same configuration as this one) and returns them
     * as a list.
     *
     * <p>If the rule does not have the specified attribute, returns the empty list.
     */
    private static Iterable<? extends TransitiveInfoCollection> getPrerequisites(
        RuleContext ruleContext, String attributeName, Mode mode) {
      if (ruleContext.getRule().isAttrDefined(attributeName, BuildType.LABEL_LIST)) {
        return ruleContext.getPrerequisites(attributeName, mode);
      } else {
        return Collections.emptyList();
      }
    }
  }
}
