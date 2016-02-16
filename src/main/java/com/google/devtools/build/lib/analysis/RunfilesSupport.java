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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.SourceManifestAction.ManifestType;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
public class RunfilesSupport {
  private static final String RUNFILES_DIR_EXT = ".runfiles";

  private final Runfiles runfiles;

  private final Artifact runfilesInputManifest;
  private final Artifact runfilesManifest;
  private final Artifact runfilesMiddleman;
  private final Artifact sourcesManifest;
  private final Artifact owningExecutable;
  private final boolean createSymlinks;
  private final ImmutableList<String> args;

  /**
   * Creates the RunfilesSupport helper with the given executable and runfiles.
   *
   * @param ruleContext the rule context to create the runfiles support for
   * @param executable the executable for whose runfiles this runfiles support is responsible, may
   *        be null
   * @param runfiles the runfiles
   * @param appendingArgs to be added after the rule's args
   */
  private RunfilesSupport(RuleContext ruleContext, Artifact executable, Runfiles runfiles,
      List<String> appendingArgs, boolean createSymlinks) {
    owningExecutable = Preconditions.checkNotNull(executable);
    this.createSymlinks = createSymlinks;

    // Adding run_under target to the runfiles manifest so it would become part
    // of runfiles tree and would be executable everywhere.
    RunUnder runUnder = ruleContext.getConfiguration().getRunUnder();
    if (runUnder != null && runUnder.getLabel() != null
        && TargetUtils.isTestRule(ruleContext.getRule())) {
      TransitiveInfoCollection runUnderTarget =
          ruleContext.getPrerequisite(":run_under", Mode.DATA);
      runfiles = new Runfiles.Builder(ruleContext.getWorkspaceName())
          .merge(getRunfiles(runUnderTarget))
          .merge(runfiles)
          .build();
    }
    this.runfiles = runfiles;

    Preconditions.checkState(!runfiles.isEmpty());

    Map<PathFragment, Artifact> symlinks = getRunfilesSymlinks();
    if (executable != null && !symlinks.containsValue(executable)) {
      throw new IllegalStateException("main program " + executable + " not included in runfiles");
    }

    Artifact artifactsMiddleman = createArtifactsMiddleman(ruleContext, runfiles.getAllArtifacts());
    runfilesInputManifest = createRunfilesInputManifestArtifact(ruleContext);
    this.runfilesManifest = createRunfilesAction(ruleContext, runfiles, artifactsMiddleman);
    this.runfilesMiddleman = createRunfilesMiddleman(
        ruleContext, artifactsMiddleman, runfilesManifest);
    sourcesManifest = createSourceManifest(ruleContext, runfiles);

    args = ImmutableList.<String>builder()
        .addAll(ruleContext.getTokenizedStringListAttr("args"))
        .addAll(appendingArgs)
        .build();
  }

  /**
   * Returns the executable owning this RunfilesSupport. Only use from Skylark.
   */
  public Artifact getExecutable() {
    return owningExecutable;
  }

  /**
   * Returns the exec path of the directory where the runfiles contained in this
   * RunfilesSupport are generated. When the owning rule has no executable,
   * returns null.
   */
  public PathFragment getRunfilesDirectoryExecPath() {
    PathFragment executablePath = owningExecutable.getExecPath();
    return executablePath.getParentDirectory().getChild(
        executablePath.getBaseName() + RUNFILES_DIR_EXT);
  }

  /** @return whether or not runfiles symlinks should be created */
  public boolean getCreateSymlinks() {
    return createSymlinks;
  }

  public Runfiles getRunfiles() {
    return runfiles;
  }

  /**
   * For executable programs, the .runfiles_manifest file outside of the
   * runfiles symlink farm; otherwise, returns null.
   *
   * <p>The MANIFEST file represents the contents of all of the symlinks in the
   * symlink farm. For efficiency, Blaze's dependency analysis ignores the
   * actual symlinks and just looks at the MANIFEST file. It is an invariant
   * that the MANIFEST file should accurately represent the contents of the
   * symlinks whenever the MANIFEST file is present.
   */
  public Artifact getRunfilesInputManifest() {
    return runfilesInputManifest;
  }

  private Artifact createRunfilesInputManifestArtifact(RuleContext context) {
    // The executable may be null for emptyRunfiles
    PathFragment relativePath = (owningExecutable != null)
        ? owningExecutable.getRootRelativePath()
        : context.getPackageDirectory().getRelative(context.getLabel().getName());
    String basename = relativePath.getBaseName();
    PathFragment inputManifestPath = relativePath.replaceName(basename + ".runfiles_manifest");
    return context.getDerivedArtifact(inputManifestPath,
        context.getConfiguration().getBinDirectory());
  }

  /**
   * For executable programs, returns the MANIFEST file in the runfiles
   * symlink farm, if blaze is run with --build_runfile_links; returns
   * the .runfiles_manifest file outside of the symlink farm, if blaze
   * is run with --nobuild_runfile_links.
   * <p>
   * Beware: In most cases {@link #getRunfilesInputManifest} is the more
   * appropriate function.
   */
  public Artifact getRunfilesManifest() {
    return runfilesManifest;
  }

  /**
   * For executable programs, the root directory of the runfiles symlink farm;
   * otherwise, returns null.
   */
  public Path getRunfilesDirectory() {
    return FileSystemUtils.replaceExtension(getRunfilesInputManifest().getPath(), RUNFILES_DIR_EXT);
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

  /**
   * Returns both runfiles artifacts and "conditional" artifacts that may be part of a
   * Runfiles PruningManifest. This means the returned set may be an overapproximation of the
   * actual set of runfiles (see {@link Runfiles.PruningManifest}).
   */
  public Iterable<Artifact> getRunfilesArtifactsWithoutMiddlemen() {
    return runfiles.getArtifactsWithoutMiddlemen();
  }

  /**
   * Returns the middleman artifact that depends on getExecutable(),
   * getRunfilesManifest(), and getRunfilesSymlinkTargets(). Anything which
   * needs to actually run the executable should depend on this.
   */
  public Artifact getRunfilesMiddleman() {
    return runfilesMiddleman;
  }

  /**
   * Returns the Sources manifest.
   */
  public Artifact getSourceManifest() {
    return sourcesManifest;
  }

  private Artifact createArtifactsMiddleman(ActionConstructionContext context,
      Iterable<Artifact> allRunfilesArtifacts) {
    return context.getAnalysisEnvironment().getMiddlemanFactory().createRunfilesMiddleman(
        context.getActionOwner(), owningExecutable, allRunfilesArtifacts,
        context.getConfiguration().getMiddlemanDirectory(), "runfiles_artifacts");
  }

  private Artifact createRunfilesMiddleman(ActionConstructionContext context,
      Artifact artifactsMiddleman, Artifact outputManifest) {
    return context.getAnalysisEnvironment().getMiddlemanFactory().createRunfilesMiddleman(
        context.getActionOwner(), owningExecutable,
        ImmutableList.of(artifactsMiddleman, outputManifest),
        context.getConfiguration().getMiddlemanDirectory(), "runfiles");
  }

  /**
   * Creates a runfiles action for all of the specified files, and returns the
   * output artifact (the artifact for the MANIFEST file).
   *
   * <p>The "runfiles" action creates a symlink farm that links all the runfiles
   * (which may come from different places, e.g. different package paths,
   * generated files, etc.) into a single tree, so that programs can access them
   * using the workspace-relative name.
   */
  private Artifact createRunfilesAction(ActionConstructionContext context, Runfiles runfiles,
      Artifact artifactsMiddleman) {
    // Compute the names of the runfiles directory and its MANIFEST file.
    Artifact inputManifest = getRunfilesInputManifest();
    context.getAnalysisEnvironment().registerAction(
        SourceManifestAction.forRunfiles(
            ManifestType.SOURCE_SYMLINKS, context.getActionOwner(), inputManifest, runfiles));

    if (!createSymlinks) {
      // Just return the manifest if that's all the build calls for.
      return inputManifest;
    }

    PathFragment runfilesDir = FileSystemUtils.replaceExtension(inputManifest.getRootRelativePath(),
        RUNFILES_DIR_EXT);
    PathFragment outputManifestPath = runfilesDir.getRelative("MANIFEST");

    BuildConfiguration config = context.getConfiguration();
    Artifact outputManifest = context.getDerivedArtifact(
        outputManifestPath, config.getBinDirectory());
    context
        .getAnalysisEnvironment()
        .registerAction(
            new SymlinkTreeAction(
                context.getActionOwner(),
                inputManifest,
                artifactsMiddleman,
                outputManifest,
                /*filesetTree=*/ false,
                config.getShExecutable()));
    return outputManifest;
  }

  /**
   * Creates an Artifact which writes the "sources only" manifest file.
   *
   * @param context the owner for the manifest action
   * @param runfiles the runfiles
   * @return the Artifact representing the file write action.
   */
  private Artifact createSourceManifest(ActionConstructionContext context, Runfiles runfiles) {
    // Put the sources only manifest next to the MANIFEST file but call it SOURCES.
    PathFragment executablePath = owningExecutable.getRootRelativePath();
    PathFragment sourcesManifestPath = executablePath.getParentDirectory().getChild(
        executablePath.getBaseName() + ".runfiles.SOURCES");
    Artifact sourceOnlyManifest = context.getDerivedArtifact(
        sourcesManifestPath, context.getConfiguration().getBinDirectory());
    context.getAnalysisEnvironment().registerAction(SourceManifestAction.forRunfiles(
        ManifestType.SOURCES_ONLY, context.getActionOwner(), sourceOnlyManifest, runfiles));
    return sourceOnlyManifest;
  }

  /**
   * Helper method that returns a collection of artifacts that are necessary for the runfiles of the
   * given target. Note that the runfile symlink tree is never built, so this may include artifacts
   * that end up not being used (see {@link Runfiles}).
   *
   * @return the Runfiles object
   */

  private static Runfiles getRunfiles(TransitiveInfoCollection target) {
    RunfilesProvider runfilesProvider = target.getProvider(RunfilesProvider.class);
    if (runfilesProvider != null) {
      return runfilesProvider.getDefaultRunfiles();
    } else {
      return Runfiles.EMPTY;
    }
  }

  /**
   * Returns the unmodifiable list of expanded and tokenized 'args' attribute
   * values.
   */
  public List<String> getArgs() {
    return args;
  }

  /**
   * Creates and returns a RunfilesSupport object for the given rule and executable. Note that this
   * method calls back into the passed in rule to obtain the runfiles.
   */
  public static RunfilesSupport withExecutable(RuleContext ruleContext, Runfiles runfiles,
      Artifact executable) {
    return new RunfilesSupport(ruleContext, executable, runfiles, ImmutableList.<String>of(),
        ruleContext.shouldCreateRunfilesSymlinks());
  }

  /**
   * Creates and returns a RunfilesSupport object for the given rule and executable. Note that this
   * method calls back into the passed in rule to obtain the runfiles.
   */
  public static RunfilesSupport withExecutable(RuleContext ruleContext, Runfiles runfiles,
      Artifact executable, boolean createSymlinks) {
    return new RunfilesSupport(ruleContext, executable, runfiles, ImmutableList.<String>of(),
        createSymlinks);
  }

  /**
   * Creates and returns a RunfilesSupport object for the given rule, executable, runfiles and args.
   */
  public static RunfilesSupport withExecutable(RuleContext ruleContext, Runfiles runfiles,
      Artifact executable, List<String> appendingArgs) {
    return new RunfilesSupport(ruleContext, executable, runfiles,
        ImmutableList.copyOf(appendingArgs), ruleContext.shouldCreateRunfilesSymlinks());
  }
}
