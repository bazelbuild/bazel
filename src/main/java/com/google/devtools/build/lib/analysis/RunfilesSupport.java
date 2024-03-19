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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.analysis.SourceManifestAction.ManifestType;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.analysis.test.TestActionBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.lang.ref.WeakReference;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * This class manages the creation of the runfiles symlink farms.
 *
 * <p>For executables that might depend on the existence of files at run-time, we create a symlink
 * farm: a directory which contains symlinks to the right locations for those runfiles.
 *
 * <p>The runfiles symlink farm serves two purposes. The first is to allow programs (and
 * programmers) to refer to files using their workspace-relative paths, regardless of whether the
 * files were source files or generated files, and regardless of which part of the package path they
 * came from. The second purpose is to ensure that all run-time dependencies are explicitly declared
 * in the BUILD files; programs may only use files which the build system knows that they depend on.
 *
 * <p>The symlink farm contains a MANIFEST file which describes its contents. The MANIFEST file
 * lists the names and contents of all of the symlinks in the symlink farm. For efficiency, Blaze's
 * dependency analysis ignores the actual symlinks and just looks at the MANIFEST file. It is an
 * invariant that the MANIFEST file should accurately represent the contents of the symlinks
 * whenever the MANIFEST file is present. build_runfile_links.py preserves this invariant (modulo
 * bugs - currently it has a bug where it may fail to preserve that invariant if it gets
 * interrupted). So the Blaze dependency analysis looks only at the MANIFEST file, rather than at
 * the individual symlinks.
 *
 * <p>We create an Artifact for the MANIFEST file and a RunfilesAction Action to create it. This
 * action does not depend on any other Artifacts.
 *
 * <p>When building an executable and running it, there are three things which must be built: the
 * executable itself, the runfiles symlink farm (represented in the action graph by the Artifact for
 * its MANIFEST), and the files pointed to by the symlinks in the symlink farm. To avoid redundancy
 * in the dependency analysis, we create a Middleman Artifact which depends on all of these. Actions
 * which will run an executable should depend on this Middleman Artifact.
 */
@Immutable
public final class RunfilesSupport {
  private static final String RUNFILES_DIR_EXT = ".runfiles";
  private static final String INPUT_MANIFEST_EXT = ".runfiles_manifest";
  private static final String OUTPUT_MANIFEST_BASENAME = "MANIFEST";
  private static final String REPO_MAPPING_MANIFEST_EXT = ".repo_mapping";

  private static class RunfilesTreeImpl implements RunfilesTree {

    private static final WeakReference<Map<PathFragment, Artifact>> NOT_YET_COMPUTED =
        new WeakReference<>(null);

    private final PathFragment execPath;
    private final Runfiles runfiles;
    private final Artifact repoMappingManifest;

    /**
     * The cached runfiles mapping. Possible values:
     *
     * <ul>
     *   <li>null if caching is not desired
     *   <li>A weak reference pointing to null if the cached value is not available (either {@link
     *       #NOT_YET_COMPUTED} or flushed from RAM)
     *   <li>A weak reference to the cached value
     * </ul>
     *
     * <p>Using weak references is preferable to soft references because {@link
     * com.google.devtools.build.lib.runtime.GcThrashingDetector} may throw a manual OOM before all
     * soft references are collected. See b/322474776.
     */
    @Nullable private volatile WeakReference<Map<PathFragment, Artifact>> cachedMapping;

    private final boolean buildRunfileLinks;
    private final RunfileSymlinksMode runfileSymlinksMode;

    private RunfilesTreeImpl(
        PathFragment execPath,
        Runfiles runfiles,
        Artifact repoMappingManifest,
        boolean buildRunfileLinks,
        boolean cacheMapping,
        RunfileSymlinksMode runfileSymlinksMode) {
      this.execPath = execPath;
      this.runfiles = runfiles;
      this.repoMappingManifest = repoMappingManifest;
      this.buildRunfileLinks = buildRunfileLinks;
      this.runfileSymlinksMode = runfileSymlinksMode;
      this.cachedMapping = cacheMapping ? NOT_YET_COMPUTED : null;
    }

    @Override
    public PathFragment getExecPath() {
      return execPath;
    }

    @Override
    public Map<PathFragment, Artifact> getMapping() {
      if (cachedMapping == null) {
        return runfiles.getRunfilesInputs(
            /* eventHandler= */ null, /* location= */ null, repoMappingManifest);
      }

      Map<PathFragment, Artifact> result = cachedMapping.get();
      if (result != null) {
        return result;
      }

      synchronized (this) {
        result = cachedMapping.get();
        if (result != null) {
          return result;
        }

        result =
            runfiles.getRunfilesInputs(
                /* eventHandler= */ null, /* location= */ null, repoMappingManifest);
        cachedMapping = new WeakReference<>(result);
        return result;
      }
    }

    @Override
    public NestedSet<Artifact> getArtifacts() {
      return runfiles.getAllArtifacts();
    }

    @Override
    public RunfileSymlinksMode getSymlinksMode() {
      return runfileSymlinksMode;
    }

    @Override
    public boolean isBuildRunfileLinks() {
      return buildRunfileLinks;
    }

    @Override
    public String getWorkspaceName() {
      return runfiles.getSuffix().getPathString();
    }
  }

  private final RunfilesTreeImpl runfilesTree;

  private final Artifact runfilesInputManifest;
  private final Artifact runfilesManifest;
  private final Artifact runfilesMiddleman;
  private final Artifact owningExecutable;
  private final CommandLine args;
  private final ActionEnvironment actionEnvironment;

  // Only cache runfiles if there is more than one test runner action. Otherwise, there is no chance
  // for reusing the runfiles within a single build, so don't pay the overhead of a weak reference.
  private static boolean cacheRunfilesMappings(RuleContext ruleContext) {
    if (!TargetUtils.isTestRule(ruleContext.getTarget())) {
      return false;
    }

    if (TestActionBuilder.getRunsPerTest(ruleContext) > 1) {
      return true;
    }

    if (TestActionBuilder.getShardCount(ruleContext) > 1) {
      return true;
    }

    return false;
  }

  /**
   * Creates the RunfilesSupport helper with the given executable and runfiles.
   *
   * @param ruleContext the rule context to create the runfiles support for
   * @param executable the executable for whose runfiles this runfiles support is responsible
   * @param runfiles the runfiles
   */
  private static RunfilesSupport create(
      RuleContext ruleContext,
      Artifact executable,
      Runfiles runfiles,
      CommandLine args,
      ActionEnvironment actionEnvironment) {
    Artifact owningExecutable = Preconditions.checkNotNull(executable);
    RunfileSymlinksMode runfileSymlinksMode =
        ruleContext.getConfiguration().getRunfileSymlinksMode();
    boolean buildRunfileManifests = ruleContext.getConfiguration().buildRunfileManifests();
    boolean buildRunfileLinks = ruleContext.getConfiguration().buildRunfileLinks();

    // Adding run_under target to the runfiles manifest so it would become part
    // of runfiles tree and would be executable everywhere.
    RunUnder runUnder = ruleContext.getConfiguration().getRunUnder();
    if (runUnder != null
        && runUnder.getLabel() != null
        && TargetUtils.isTestRule(ruleContext.getRule())) {
      TransitiveInfoCollection runUnderTarget = ruleContext.getPrerequisite(":run_under");
      runfiles =
          new Runfiles.Builder(
                  ruleContext.getWorkspaceName(),
                  ruleContext.getConfiguration().legacyExternalRunfiles())
              .merge(getRunfiles(runUnderTarget, ruleContext.getWorkspaceName()))
              .merge(runfiles)
              .build();
    }
    Preconditions.checkState(!runfiles.isEmpty());

    Artifact repoMappingManifest =
        createRepoMappingManifestAction(ruleContext, runfiles, owningExecutable);

    Artifact runfilesInputManifest;
    Artifact runfilesManifest;
    if (buildRunfileManifests) {
      runfilesInputManifest = createRunfilesInputManifestArtifact(ruleContext, owningExecutable);
      runfilesManifest =
          createRunfilesAction(
              ruleContext, runfiles, buildRunfileLinks, runfilesInputManifest, repoMappingManifest);
    } else {
      runfilesInputManifest = null;
      runfilesManifest = null;
    }

    PathFragment executablePath = owningExecutable.getExecPath();
    PathFragment runfilesExecPath =
        executablePath.replaceName(executablePath.getBaseName() + RUNFILES_DIR_EXT);

    RunfilesTreeImpl runfilesTree =
        new RunfilesTreeImpl(
            runfilesExecPath,
            runfiles,
            repoMappingManifest,
            buildRunfileLinks,
            cacheRunfilesMappings(ruleContext),
            runfileSymlinksMode);

    Artifact runfilesMiddleman =
        createRunfilesMiddleman(
            ruleContext, owningExecutable, runfilesTree, runfilesManifest, repoMappingManifest);

    return new RunfilesSupport(
        runfilesTree,
        runfilesInputManifest,
        runfilesManifest,
        runfilesMiddleman,
        owningExecutable,
        args,
        actionEnvironment);
  }

  private RunfilesSupport(
      RunfilesTreeImpl runfilesTree,
      Artifact runfilesInputManifest,
      Artifact runfilesManifest,
      Artifact runfilesMiddleman,
      Artifact owningExecutable,
      CommandLine args,
      ActionEnvironment actionEnvironment) {
    this.runfilesTree = runfilesTree;
    this.runfilesInputManifest = runfilesInputManifest;
    this.runfilesManifest = runfilesManifest;
    this.runfilesMiddleman = runfilesMiddleman;
    this.owningExecutable = owningExecutable;
    this.args = args;
    this.actionEnvironment = actionEnvironment;
  }

  /** Returns the executable owning this RunfilesSupport. */
  public Artifact getExecutable() {
    return owningExecutable;
  }

  public Runfiles getRunfiles() {
    return runfilesTree.runfiles;
  }

  /**
   * Helper method that returns a collection of artifacts that are necessary for the runfiles of the
   * given target. Note that the runfile symlink tree is never built, so this may include artifacts
   * that end up not being used (see {@link Runfiles}).
   *
   * @return the Runfiles object
   */
  private static Runfiles getRunfiles(TransitiveInfoCollection target, String workspaceName) {
    RunfilesProvider runfilesProvider = target.getProvider(RunfilesProvider.class);
    if (runfilesProvider != null) {
      return runfilesProvider.getDefaultRunfiles();
    } else {
      return new Runfiles.Builder(workspaceName)
          .addTransitiveArtifacts(target.getProvider(FileProvider.class).getFilesToBuild())
          .build();
    }
  }

  /**
   * Returns the .runfiles_manifest file outside of the runfiles symlink farm. Returns null if
   * --nobuild_runfile_manifests is in effect.
   *
   * <p>The MANIFEST file represents the contents of all of the symlinks in the symlink farm. For
   * efficiency, Blaze's dependency analysis ignores the actual symlinks and just looks at the
   * MANIFEST file. It is an invariant that the MANIFEST file should accurately represent the
   * contents of the symlinks whenever the MANIFEST file is present.
   */
  @Nullable
  public Artifact getRunfilesInputManifest() {
    return runfilesInputManifest;
  }

  private static Artifact createRunfilesInputManifestArtifact(
      RuleContext context, Artifact owningExecutable) {
    // The executable may be null for emptyRunfiles
    PathFragment relativePath =
        (owningExecutable != null)
            ? owningExecutable.getOutputDirRelativePath(
                context.getConfiguration().isSiblingRepositoryLayout())
            : context.getPackageDirectory().getRelative(context.getLabel().getName());
    String basename = relativePath.getBaseName();
    PathFragment inputManifestPath = relativePath.replaceName(basename + INPUT_MANIFEST_EXT);
    return context.getDerivedArtifact(inputManifestPath, context.getBinDirectory());
  }

  /**
   * Returns the MANIFEST file in the runfiles symlink farm if Bazel is run with
   * --build_runfile_links. Returns the .runfiles_manifest file outside of the symlink farm, if
   * Bazel is run with --nobuild_runfile_links. Returns null if --nobuild_runfile_manifests is
   * passed.
   *
   * <p>Beware: In most cases {@link #getRunfilesInputManifest} is the more appropriate function.
   */
  @Nullable
  public Artifact getRunfilesManifest() {
    return runfilesManifest;
  }

  /**
   * Returns the foo.repo_mapping file if Bazel is run with transitive package tracking turned on
   * (see {@code SkyframeExecutor#getForcedSingleSourceRootIfNoExecrootSymlinkCreation}) and any of
   * the transitive packages come from a repository with strict deps (see {@code
   * #collectRepoMappings}). Otherwise, returns null.
   */
  @Nullable
  public Artifact getRepoMappingManifest() {
    return runfilesTree.repoMappingManifest;
  }

  /** Returns the root directory of the runfiles symlink farm; otherwise, returns null. */
  @Nullable
  public Path getRunfilesDirectory() {
    if (runfilesInputManifest == null) {
      return null;
    }
    return FileSystemUtils.replaceExtension(runfilesInputManifest.getPath(), RUNFILES_DIR_EXT);
  }

  /**
   * Returns the files pointed to by the symlinks in the runfiles symlink farm. This method is slow.
   */
  @VisibleForTesting
  public Collection<Artifact> getRunfilesSymlinkTargets() {
    return getRunfilesSymlinks().values();
  }

  /**
   * Returns the names of the symlinks in the runfiles symlink farm as a Set of PathFragments. This
   * method is slow.
   */
  // We should make this VisibleForTesting, but it is still used by TestHelper
  public Set<PathFragment> getRunfilesSymlinkNames() {
    return getRunfilesSymlinks().keySet();
  }

  /**
   * Returns the names of the symlinks in the runfiles symlink farm as a Set of PathFragments. This
   * method is slow.
   */
  @VisibleForTesting
  public Map<PathFragment, Artifact> getRunfilesSymlinks() {
    return runfilesTree.runfiles.asMapWithoutRootSymlinks();
  }

  /**
   * Returns the middleman artifact that depends on getExecutable(), getRunfilesManifest(), and
   * getRunfilesSymlinkTargets(). Anything which needs to actually run the executable should depend
   * on this.
   */
  public Artifact getRunfilesMiddleman() {
    return runfilesMiddleman;
  }

  private static Artifact createRunfilesMiddleman(
      ActionConstructionContext context,
      Artifact owningExecutable,
      RunfilesTree runfilesTree,
      @Nullable Artifact runfilesManifest,
      Artifact repoMappingManifest) {
    return context
        .getAnalysisEnvironment()
        .getMiddlemanFactory()
        .createRunfilesMiddleman(
            context.getActionOwner(),
            owningExecutable,
            runfilesTree,
            runfilesManifest,
            repoMappingManifest,
            context.getMiddlemanDirectory());
  }

  /**
   * Creates a runfiles action for all of the specified files, and returns the output artifact (the
   * artifact for the MANIFEST file).
   *
   * <p>The "runfiles" action creates a symlink farm that links all the runfiles (which may come
   * from different places, e.g. different package paths, generated files, etc.) into a single tree,
   * so that programs can access them using the workspace-relative name.
   */
  private static Artifact createRunfilesAction(
      ActionConstructionContext context,
      Runfiles runfiles,
      boolean createSymlinks,
      Artifact inputManifest,
      @Nullable Artifact repoMappingManifest) {
    // Compute the names of the runfiles directory and its MANIFEST file.
    context
        .getAnalysisEnvironment()
        .registerAction(
            new SourceManifestAction(
                ManifestType.SOURCE_SYMLINKS,
                context.getActionOwner(),
                inputManifest,
                runfiles,
                repoMappingManifest,
                context.getConfiguration().remotableSourceManifestActions()));

    if (!createSymlinks) {
      // Just return the manifest if that's all the build calls for.
      return inputManifest;
    }

    PathFragment runfilesDir =
        FileSystemUtils.replaceExtension(
            inputManifest.getOutputDirRelativePath(
                context.getConfiguration().isSiblingRepositoryLayout()),
            RUNFILES_DIR_EXT);
    PathFragment outputManifestPath = runfilesDir.getRelative(OUTPUT_MANIFEST_BASENAME);

    BuildConfigurationValue config = context.getConfiguration();
    Artifact outputManifest =
        context.getDerivedArtifact(outputManifestPath, context.getBinDirectory());
    context
        .getAnalysisEnvironment()
        .registerAction(
            new SymlinkTreeAction(
                context.getActionOwner(),
                config,
                inputManifest,
                runfiles,
                outputManifest,
                repoMappingManifest,
                /*filesetRoot=*/ null));
    return outputManifest;
  }

  /** Returns the unmodifiable list of expanded and tokenized 'args' attribute values. */
  public CommandLine getArgs() {
    return args;
  }

  /** Returns the immutable environment from the 'env' and 'env_inherit' attribute values. */
  public ActionEnvironment getActionEnvironment() {
    return actionEnvironment;
  }

  /**
   * Creates and returns a {@link RunfilesSupport} object for the given rule and executable. Note
   * that this method calls back into the passed in rule to obtain the runfiles.
   *
   * <p>If the executable is a test, runfiles mappings are cached and re-used between shards. It's a
   * win since when there is a large number of test shards and/or runs per test, the same runfiles
   * tree is needed many times.
   */
  public static RunfilesSupport withExecutable(
      RuleContext ruleContext, Runfiles runfiles, Artifact executable) throws InterruptedException {
    return RunfilesSupport.create(
        ruleContext,
        executable,
        runfiles,
        computeArgs(ruleContext),
        computeActionEnvironment(ruleContext));
  }

  private static CommandLine computeArgs(RuleContext ruleContext) throws InterruptedException {
    if (!ruleContext.getRule().isAttrDefined("args", Types.STRING_LIST)) {
      // Some non-_binary rules create RunfilesSupport instances; it is fine to not have an args
      // attribute here.
      return CommandLine.empty();
    }
    ImmutableList<String> args = ruleContext.getExpander().withDataLocations().tokenized("args");
    return args.isEmpty() ? CommandLine.empty() : CommandLine.of(args);
  }

  private static ActionEnvironment computeActionEnvironment(RuleContext ruleContext)
      throws InterruptedException {
    // Executable Starlark rules can use RunEnvironmentInfo to specify environment variables.
    boolean isNativeRule =
        ruleContext.getRule().getRuleClassObject().getRuleDefinitionEnvironmentLabel() == null;
    if (!isNativeRule
        || (!ruleContext.getRule().isAttrDefined("env", Types.STRING_DICT)
            && !ruleContext.getRule().isAttrDefined("env_inherit", Types.STRING_LIST))) {
      return ActionEnvironment.EMPTY;
    }
    TreeMap<String, String> fixedEnv = new TreeMap<>();
    Set<String> inheritedEnv = new LinkedHashSet<>();
    if (ruleContext.isAttrDefined("env", Types.STRING_DICT)) {
      Expander expander = ruleContext.getExpander().withDataLocations();
      for (Map.Entry<String, String> entry :
          ruleContext.attributes().get("env", Types.STRING_DICT).entrySet()) {
        fixedEnv.put(entry.getKey(), expander.expand("env", entry.getValue()));
      }
    }
    if (ruleContext.isAttrDefined("env_inherit", Types.STRING_LIST)) {
      for (String key : ruleContext.attributes().get("env_inherit", Types.STRING_LIST)) {
        if (!fixedEnv.containsKey(key)) {
          inheritedEnv.add(key);
        }
      }
    }
    return ActionEnvironment.create(
        ImmutableMap.copyOf(fixedEnv), ImmutableSet.copyOf(inheritedEnv));
  }

  /** Returns the path of the input manifest of {@code runfilesDir}. */
  public static Path inputManifestPath(Path runfilesDir) {
    return FileSystemUtils.replaceExtension(runfilesDir, INPUT_MANIFEST_EXT);
  }

  /** Returns the path of the output manifest of {@code runfilesDir}. */
  public static Path outputManifestPath(Path runfilesDir) {
    return runfilesDir.getRelative(OUTPUT_MANIFEST_BASENAME);
  }

  @Nullable
  private static Artifact createRepoMappingManifestAction(
      RuleContext ruleContext, Runfiles runfiles, Artifact owningExecutable) {
    if (ruleContext.getTransitivePackagesForRunfileRepoMappingManifest() == null) {
      // If transitive packages are not tracked for repo mapping manifest, we don't need the action.
      return null;
    }

    PathFragment executablePath =
        (owningExecutable != null)
            ? owningExecutable.getOutputDirRelativePath(
                ruleContext.getConfiguration().isSiblingRepositoryLayout())
            : ruleContext.getPackageDirectory().getRelative(ruleContext.getLabel().getName());
    Artifact repoMappingManifest =
        ruleContext.getDerivedArtifact(
            executablePath.replaceName(executablePath.getBaseName() + REPO_MAPPING_MANIFEST_EXT),
            ruleContext.getBinDirectory());
    ruleContext
        .getAnalysisEnvironment()
        .registerAction(
            new RepoMappingManifestAction(
                ruleContext.getActionOwner(),
                repoMappingManifest,
                ruleContext.getTransitivePackagesForRunfileRepoMappingManifest(),
                runfiles.getArtifacts(),
                runfiles.getSymlinks(),
                runfiles.getRootSymlinks(),
                ruleContext.getWorkspaceName()));
    return repoMappingManifest;
  }

  public RunfilesTree getRunfilesTree() {
    return runfilesTree;
  }
}
