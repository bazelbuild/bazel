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
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.SourceManifestAction.ManifestType;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
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
public final class RunfilesSupport implements RunfilesSupplier {
  private static final String RUNFILES_DIR_EXT = ".runfiles";
  private static final String INPUT_MANIFEST_EXT = ".runfiles_manifest";
  private static final String OUTPUT_MANIFEST_BASENAME = "MANIFEST";
  private static final String REPO_MAPPING_MANIFEST_EXT = ".repo_mapping";

  private final Runfiles runfiles;

  private final Artifact runfilesInputManifest;
  private final Artifact runfilesManifest;
  private final Artifact repoMappingManifest;
  private final Artifact runfilesMiddleman;
  private final Artifact owningExecutable;
  private final boolean buildRunfileLinks;
  private final boolean runfilesEnabled;
  private final CommandLine args;
  private final ActionEnvironment actionEnvironment;

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
    boolean createManifest = ruleContext.getConfiguration().buildRunfilesManifests();
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
    if (createManifest) {
      runfilesInputManifest = createRunfilesInputManifestArtifact(ruleContext, owningExecutable);
      runfilesManifest =
          createRunfilesAction(
              ruleContext, runfiles, buildRunfileLinks, runfilesInputManifest, repoMappingManifest);
    } else {
      runfilesInputManifest = null;
      runfilesManifest = null;
    }
    Artifact runfilesMiddleman =
        createRunfilesMiddleman(
            ruleContext, owningExecutable, runfiles, runfilesManifest, repoMappingManifest);

    boolean runfilesEnabled = ruleContext.getConfiguration().runfilesEnabled();

    return new RunfilesSupport(
        runfiles,
        runfilesInputManifest,
        runfilesManifest,
        repoMappingManifest,
        runfilesMiddleman,
        owningExecutable,
        buildRunfileLinks,
        runfilesEnabled,
        args,
        actionEnvironment);
  }

  private RunfilesSupport(
      Runfiles runfiles,
      Artifact runfilesInputManifest,
      Artifact runfilesManifest,
      Artifact repoMappingManifest,
      Artifact runfilesMiddleman,
      Artifact owningExecutable,
      boolean buildRunfileLinks,
      boolean runfilesEnabled,
      CommandLine args,
      ActionEnvironment actionEnvironment) {
    this.runfiles = runfiles;
    this.runfilesInputManifest = runfilesInputManifest;
    this.runfilesManifest = runfilesManifest;
    this.repoMappingManifest = repoMappingManifest;
    this.runfilesMiddleman = runfilesMiddleman;
    this.owningExecutable = owningExecutable;
    this.buildRunfileLinks = buildRunfileLinks;
    this.runfilesEnabled = runfilesEnabled;
    this.args = args;
    this.actionEnvironment = actionEnvironment;
  }

  /** Returns the executable owning this RunfilesSupport. Only use from Starlark. */
  public Artifact getExecutable() {
    return owningExecutable;
  }

  /** Returns the path of the runfiles directory relative to the exec root. */
  public PathFragment getRunfilesDirectoryExecPath() {
    PathFragment executablePath = owningExecutable.getExecPath();
    return executablePath.replaceName(executablePath.getBaseName() + RUNFILES_DIR_EXT);
  }

  /**
   * Returns {@code true} if runfile symlinks should be materialized when building an executable.
   *
   * <p>Also see {@link #isRunfilesEnabled()}.
   */
  public boolean isBuildRunfileLinks() {
    return buildRunfileLinks;
  }

  @Override
  public boolean isBuildRunfileLinks(PathFragment runfilesDir) {
    return buildRunfileLinks && runfilesDir.equals(getRunfilesDirectoryExecPath());
  }

  /**
   * Returns {@code true} if runfile symlinks are enabled.
   *
   * <p>This option differs from {@link #isBuildRunfileLinks()} in that if {@code false} it also
   * disables runfile symlinks creation during run/test.
   */
  public boolean isRunfilesEnabled() {
    return runfilesEnabled;
  }

  public Runfiles getRunfiles() {
    return runfiles;
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
          .addTransitiveArtifacts(target.getProvider(FilesToRunProvider.class).getFilesToRun())
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
    return repoMappingManifest;
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
    return runfiles.asMapWithoutRootSymlinks();
  }

  /** Returns the artifacts in the runfiles tree. */
  public NestedSet<Artifact> getRunfilesArtifacts() {
    return runfiles.getArtifacts();
  }

  /** Returns the name of the workspace that the build is occurring in. */
  public PathFragment getWorkspaceName() {
    return runfiles.getSuffix();
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
      Runfiles runfiles,
      @Nullable Artifact runfilesManifest,
      Artifact repoMappingManifest) {
    NestedSetBuilder<Artifact> deps = NestedSetBuilder.stableOrder();
    deps.addTransitive(runfiles.getAllArtifacts());
    if (runfilesManifest != null) {
      deps.add(runfilesManifest);
    }
    if (repoMappingManifest != null) {
      deps.add(repoMappingManifest);
    }
    return context
        .getAnalysisEnvironment()
        .getMiddlemanFactory()
        .createRunfilesMiddleman(
            context.getActionOwner(),
            owningExecutable,
            deps.build(),
            context.getMiddlemanDirectory(),
            "runfiles");
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
   */
  public static RunfilesSupport withExecutable(
      RuleContext ruleContext, Runfiles runfiles, Artifact executable) throws InterruptedException {
    return RunfilesSupport.create(
        ruleContext,
        executable,
        runfiles,
        computeArgs(ruleContext, CommandLine.EMPTY),
        computeActionEnvironment(ruleContext));
  }

  /**
   * Creates and returns a {@link RunfilesSupport} object for the given rule and executable. Note
   * that this method calls back into the passed in rule to obtain the runfiles.
   */
  public static RunfilesSupport withExecutable(
      RuleContext ruleContext, Runfiles runfiles, Artifact executable, List<String> appendingArgs)
      throws InterruptedException {
    return RunfilesSupport.create(
        ruleContext,
        executable,
        runfiles,
        computeArgs(ruleContext, CommandLine.of(appendingArgs)),
        computeActionEnvironment(ruleContext));
  }

  /**
   * Creates and returns a {@link RunfilesSupport} object for the given rule, executable, runfiles
   * and args.
   */
  public static RunfilesSupport withExecutable(
      RuleContext ruleContext, Runfiles runfiles, Artifact executable, CommandLine appendingArgs)
      throws InterruptedException {
    return RunfilesSupport.create(
        ruleContext,
        executable,
        runfiles,
        computeArgs(ruleContext, appendingArgs),
        computeActionEnvironment(ruleContext));
  }

  private static CommandLine computeArgs(RuleContext ruleContext, CommandLine additionalArgs)
      throws InterruptedException {
    if (!ruleContext.getRule().isAttrDefined("args", Type.STRING_LIST)) {
      // Some non-_binary rules create RunfilesSupport instances; it is fine to not have an args
      // attribute here.
      return additionalArgs;
    }
    return CommandLine.concat(
        ruleContext.getExpander().withDataLocations().tokenized("args"), additionalArgs);
  }

  private static ActionEnvironment computeActionEnvironment(RuleContext ruleContext)
      throws InterruptedException {
    // Executable Starlark rules can use RunEnvironmentInfo to specify environment variables.
    boolean isNativeRule =
        ruleContext.getRule().getRuleClassObject().getRuleDefinitionEnvironmentLabel() == null;
    if (!isNativeRule
        || (!ruleContext.getRule().isAttrDefined("env", Type.STRING_DICT)
            && !ruleContext.getRule().isAttrDefined("env_inherit", Type.STRING_LIST))) {
      return ActionEnvironment.EMPTY;
    }
    TreeMap<String, String> fixedEnv = new TreeMap<>();
    Set<String> inheritedEnv = new LinkedHashSet<>();
    if (ruleContext.isAttrDefined("env", Type.STRING_DICT)) {
      Expander expander = ruleContext.getExpander().withDataLocations();
      for (Map.Entry<String, String> entry :
          ruleContext.attributes().get("env", Type.STRING_DICT).entrySet()) {
        fixedEnv.put(entry.getKey(), expander.expand("env", entry.getValue()));
      }
    }
    if (ruleContext.isAttrDefined("env_inherit", Type.STRING_LIST)) {
      for (String key : ruleContext.attributes().get("env_inherit", Type.STRING_LIST)) {
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

  @Override
  public NestedSet<Artifact> getArtifacts() {
    return runfiles.getArtifacts();
  }

  @Override
  public ImmutableSet<PathFragment> getRunfilesDirs() {
    return ImmutableSet.of(getRunfilesDirectoryExecPath());
  }

  @Override
  public ImmutableMap<PathFragment, Map<PathFragment, Artifact>> getMappings() {
    return ImmutableMap.of(
        getRunfilesDirectoryExecPath(),
        runfiles.getRunfilesInputs(
            /* eventHandler= */ null, /* location= */ null, repoMappingManifest));
  }

  @Override
  public ImmutableList<Artifact> getManifests() {
    return ImmutableList.of();
  }

  @Override
  public boolean isRunfileLinksEnabled(PathFragment runfilesDir) {
    return runfilesEnabled && runfilesDir.equals(getRunfilesDirectoryExecPath());
  }

  @Override
  public RunfilesSupplier withOverriddenRunfilesDir(PathFragment newRunfilesDir) {
    return newRunfilesDir.equals(getRunfilesDirectoryExecPath())
        ? this
        : new SingleRunfilesSupplier(
            newRunfilesDir,
            runfiles,
            /* manifest= */ null,
            repoMappingManifest,
            buildRunfileLinks,
            runfilesEnabled);
  }
}
