// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.SingleRunfilesSupplier;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Key;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Options;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.WorkspaceInfoFromDiff;
import com.google.devtools.build.lib.testutil.FakeAttributeMapper;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.CrashFailureDetails;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Utilities for analysis phase tests.
 */
public final class AnalysisTestUtil {

  /** TopLevelArtifactContext that should be sufficient for testing. */
  public static final TopLevelArtifactContext TOP_LEVEL_ARTIFACT_CONTEXT =
      new TopLevelArtifactContext(
          /*runTestsExclusively=*/ false,
          /*expandFilesets=*/ false,
          /*fullyResolveFilesetSymlinks=*/ false,
          /*outputGroups=*/ ImmutableSortedSet.copyOf(OutputGroupInfo.DEFAULT_GROUPS));

  /**
   * An {@link AnalysisEnvironment} implementation that collects the actions registered.
   */
  public static class CollectingAnalysisEnvironment implements AnalysisEnvironment {
    private final List<ActionAnalysisMetadata> actions = new ArrayList<>();
    private final AnalysisEnvironment original;

    public CollectingAnalysisEnvironment(AnalysisEnvironment original) {
      this.original = original;
    }

    public void clear() {
      actions.clear();
    }

    @Override
    public void registerAction(ActionAnalysisMetadata action) {
      this.actions.add(action);
      original.registerAction(action);
    }

    /** Calls {@link MutableActionGraph#registerAction} for all collected actions. */
    public void registerWith(MutableActionGraph actionGraph) throws InterruptedException {
      for (ActionAnalysisMetadata action : actions) {
        try {
          actionGraph.registerAction(action);
        } catch (ActionConflictException e) {
          throw new ActionsTestUtil.UncheckedActionConflictException(e);
        }
      }
    }

    @Override
    public ExtendedEventHandler getEventHandler() {
      return original.getEventHandler();
    }

    @Override
    public boolean hasErrors() {
      return original.hasErrors();
    }

    @Override
    public Artifact.DerivedArtifact getDerivedArtifact(
        PathFragment rootRelativePath, ArtifactRoot root) {
      return original.getDerivedArtifact(rootRelativePath, root);
    }

    @Override
    public Artifact.DerivedArtifact getDerivedArtifact(
        PathFragment rootRelativePath, ArtifactRoot root, boolean contentBasedPath) {
      return original.getDerivedArtifact(rootRelativePath, root, contentBasedPath);
    }

    @Override
    public Artifact getConstantMetadataArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      return original.getConstantMetadataArtifact(rootRelativePath, root);
    }

    @Override
    public SpecialArtifact getTreeArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      return original.getTreeArtifact(rootRelativePath, root);
    }

    @Override
    public SpecialArtifact getSymlinkArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      return original.getSymlinkArtifact(rootRelativePath, root);
    }

    @Override
    public Artifact getFilesetArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      return original.getFilesetArtifact(rootRelativePath, root);
    }

    @Override
    public MiddlemanFactory getMiddlemanFactory() {
      return original.getMiddlemanFactory();
    }

    @Override
    public ActionAnalysisMetadata getLocalGeneratingAction(Artifact artifact) {
      return original.getLocalGeneratingAction(artifact);
    }

    @Override
    public ImmutableList<ActionAnalysisMetadata> getRegisteredActions() {
      return original.getRegisteredActions();
    }

    @Override
    public SkyFunction.Environment getSkyframeEnv() {
      return null;
    }

    @Override
    public StarlarkSemantics getStarlarkSemantics() {
      return original.getStarlarkSemantics();
    }

    @Override
    public ImmutableMap<String, Object> getStarlarkDefinedBuiltins() throws InterruptedException {
      return original.getStarlarkDefinedBuiltins();
    }

    @Override
    public Artifact getStableWorkspaceStatusArtifact() throws InterruptedException {
      return original.getStableWorkspaceStatusArtifact();
    }

    @Override
    public Artifact getVolatileWorkspaceStatusArtifact() throws InterruptedException {
      return original.getVolatileWorkspaceStatusArtifact();
    }

    @Override
    public ImmutableList<Artifact> getBuildInfo(
        boolean stamp, BuildInfoKey key, BuildConfigurationValue config)
        throws InterruptedException {
      return original.getBuildInfo(stamp, key, config);
    }

    @Override
    public ActionLookupKey getOwner() {
      return original.getOwner();
    }

    @Override
    public ImmutableSet<Artifact> getOrphanArtifacts() {
      return original.getOrphanArtifacts();
    }

    @Override
    public ImmutableSet<Artifact> getTreeArtifactsConflictingWithFiles() {
      return original.getTreeArtifactsConflictingWithFiles();
    }

    @Override
    public ActionKeyContext getActionKeyContext() {
      return original.getActionKeyContext();
    }
  }

  /** A dummy WorkspaceStatusAction. */
  @Immutable
  public static final class DummyWorkspaceStatusAction extends WorkspaceStatusAction {
    private final Artifact stableStatus;
    private final Artifact volatileStatus;

    public DummyWorkspaceStatusAction(Artifact stableStatus, Artifact volatileStatus) {
      super(
          ActionOwner.SYSTEM_ACTION_OWNER,
          NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          ImmutableSet.of(stableStatus, volatileStatus),
          "workspace status");
      this.stableStatus = stableStatus;
      this.volatileStatus = volatileStatus;
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      try {
        FileSystemUtils.writeContent(
            actionExecutionContext.getInputPath(stableStatus), new byte[] {});
        FileSystemUtils.writeContent(
            actionExecutionContext.getInputPath(volatileStatus), new byte[] {});
      } catch (IOException e) {
        throw new ActionExecutionException(
            e, this, true, CrashFailureDetails.detailedExitCodeForThrowable(e));
      }
      return ActionResult.EMPTY;
    }

    @Override
    public String getMnemonic() {
      return "DummyBuildInfoAction";
    }

    @Override
    public void computeKey(
        ActionKeyContext actionKeyContext,
        @Nullable Artifact.ArtifactExpander artifactExpander,
        Fingerprint fp) {}

    @Override
    public Artifact getVolatileStatus() {
      return volatileStatus;
    }

    @Override
    public Artifact getStableStatus() {
      return stableStatus;
    }
  }

  /** A WorkspaceStatusAction.Context that has no stable keys and no volatile keys. */
  public static class DummyWorkspaceStatusActionContext implements WorkspaceStatusAction.Context {
    @Override
    public ImmutableMap<String, Key> getStableKeys() {
      return ImmutableMap.of();
    }

    @Override
    public ImmutableMap<String, Key> getVolatileKeys() {
      return ImmutableMap.of();
    }

    @Override
    public Options getOptions() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ImmutableMap<String, String> getClientEnv() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Command getCommand() {
      throw new UnsupportedOperationException();
    }
  }

  /** A workspace status action factory that does not do any interaction with the environment. */
  public static final class DummyWorkspaceStatusActionFactory
      implements WorkspaceStatusAction.Factory {

    @Override
    public WorkspaceStatusAction createWorkspaceStatusAction(
        WorkspaceStatusAction.Environment env) {
      Artifact stableStatus = env.createStableArtifact("build-info.txt");
      Artifact volatileStatus = env.createVolatileArtifact("build-changelist.txt");
      return new DummyWorkspaceStatusAction(stableStatus, volatileStatus);
    }

    @Override
    public ImmutableSortedMap<String, String> createDummyWorkspaceStatus(
        @Nullable WorkspaceInfoFromDiff workspaceInfoFromDiff) {
      return ImmutableSortedMap.of();
    }
  }

  public static final AnalysisEnvironment STUB_ANALYSIS_ENVIRONMENT = new StubAnalysisEnvironment();

  /** An AnalysisEnvironment with stubbed-out methods. */
  public static class StubAnalysisEnvironment implements AnalysisEnvironment {
    private static final ActionLookupKey DUMMY_KEY =
        new ActionLookupKey() {
          @Nullable
          @Override
          public Label getLabel() {
            return null;
          }

          @Nullable
          @Override
          public BuildConfigurationKey getConfigurationKey() {
            return null;
          }

          @Override
          public SkyFunctionName functionName() {
            return null;
          }
        };

    @Override
    public void registerAction(ActionAnalysisMetadata action) {}

    @Override
    public boolean hasErrors() {
      return false;
    }

    @Override
    public Artifact getConstantMetadataArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      return null;
    }

    @Override
    public SpecialArtifact getTreeArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      return null;
    }

    @Override
    public SpecialArtifact getSymlinkArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      return null;
    }

    @Override
    public ExtendedEventHandler getEventHandler() {
      return null;
    }

    @Override
    public MiddlemanFactory getMiddlemanFactory() {
      return null;
    }

    @Override
    public Action getLocalGeneratingAction(Artifact artifact) {
      return null;
    }

    @Override
    public ImmutableList<ActionAnalysisMetadata> getRegisteredActions() {
      return ImmutableList.of();
    }

    @Override
    public SkyFunction.Environment getSkyframeEnv() {
      return null;
    }

    @Override
    public StarlarkSemantics getStarlarkSemantics() {
      return null;
    }

    @Override
    public ImmutableMap<String, Object> getStarlarkDefinedBuiltins() throws InterruptedException {
      return null;
    }

    @Override
    public Artifact getFilesetArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      return null;
    }

    @Override
    public Artifact.DerivedArtifact getDerivedArtifact(
        PathFragment rootRelativePath, ArtifactRoot root) {
      return null;
    }

    @Override
    public Artifact.DerivedArtifact getDerivedArtifact(
        PathFragment rootRelativePath, ArtifactRoot root, boolean contentBasedPath) {
      return null;
    }

    @Override
    public Artifact getStableWorkspaceStatusArtifact() {
      return null;
    }

    @Override
    public Artifact getVolatileWorkspaceStatusArtifact() {
      return null;
    }

    @Override
    public ImmutableList<Artifact> getBuildInfo(
        boolean stamp, BuildInfoKey key, BuildConfigurationValue config) {
      return ImmutableList.of();
    }

    @Override
    public ActionLookupKey getOwner() {
      return DUMMY_KEY;
    }

    @Override
    public ImmutableSet<Artifact> getOrphanArtifacts() {
      return ImmutableSet.of();
    }

    @Override
    public ImmutableSet<Artifact> getTreeArtifactsConflictingWithFiles() {
      return ImmutableSet.of();
    }

    @Override
    public ActionKeyContext getActionKeyContext() {
      return null;
    }
  }

  /** Matches the output path prefix contributed by a C++ configuration fragment. */
  private static final Pattern OUTPUT_PATH_CPP_PREFIX_PATTERN =
      Pattern.compile("(?<=" + TestConstants.PRODUCT_NAME + "-out/)gcc[^/]*-grte-\\w+-");

  /** Matches the output path prefix contributed by an Android configuration fragment. */
  private static final Pattern OUTPUT_PATH_ANDROID_PREFIX_PATTERN =
      Pattern.compile("(?<=" + TestConstants.PRODUCT_NAME + "-out/)android-");

  /**
   * Apply {@code function} to the path string of the given ArtifactRoot. If the root path matches
   * {@link OUTPUT_PATH_CPP_PREFIX_PATTERN} or {@link OUTPUT_PATH_ANDROID_PREFIX_PATTERN}, also use
   * those to update the path and invoke {@code function} again.
   *
   * @return the result of {@code function} from the most specific root path
   */
  private static <U> U computeRootPaths(ArtifactRoot artifactRoot, Function<String, U> function) {
    String rootPath = artifactRoot.getRoot().toString();
    U result = function.apply(rootPath);
    // The output paths that bin, genfiles, etc. refer to may or may not include the C++-contributed
    // pieces. e.g. they may be bazel-out/gcc-X-glibc-Y-k8-fastbuild/ or they may be
    // bazel-out/fastbuild/. This code adds support for the non-C++ case, too.
    String cppReplacedPath = OUTPUT_PATH_CPP_PREFIX_PATTERN.matcher(rootPath).replaceFirst("");
    if (!rootPath.equals(cppReplacedPath)) {
      result = function.apply(cppReplacedPath);
    }
    // Also handle Android output paths in the same way.
    String androidReplacedPath =
        OUTPUT_PATH_ANDROID_PREFIX_PATTERN.matcher(rootPath).replaceFirst("");
    if (!rootPath.equals(androidReplacedPath)) {
      result = function.apply(androidReplacedPath);
    }
    return result;
  }

  /**
   * Given a collection of Artifacts, returns a corresponding set of strings of the form "{root}
   * {relpath}", such as "bin x/libx.a". Such strings make assertions easier to write.
   *
   * <p>The returned set preserves the order of the input.
   */
  public static Set<String> artifactsToStrings(
      BuildConfigurationValue targetConfiguration,
      Iterable<? extends Artifact> artifacts) {
    Map<String, String> rootMap = new HashMap<>();
    computeRootPaths(
        targetConfiguration.getBinDirectory(RepositoryName.MAIN), path -> rootMap.put(path, "bin"));
    // In preparation for merging genfiles/ and bin/, we don't differentiate them in tests anymore
    computeRootPaths(
        targetConfiguration.getGenfilesDirectory(RepositoryName.MAIN),
        path -> rootMap.put(path, "bin"));
    computeRootPaths(
        targetConfiguration.getMiddlemanDirectory(RepositoryName.MAIN),
        path -> rootMap.put(path, "internal"));

    Set<String> files = new LinkedHashSet<>();
    for (Artifact artifact : artifacts) {
      ArtifactRoot root = artifact.getRoot();
      if (root.isSourceRoot()) {
        files.add("src " + artifact.getExecPath());
      } else {
        // Find the most specific mapping.
        String name = computeRootPaths(root, path -> rootMap.getOrDefault(path, "/"));
        files.add(name + " " + artifact.getRootRelativePath());
      }
    }
    return files;
  }

  /** Creates a {@link RunfilesSupplier} for use in tests. */
  public static RunfilesSupplier createRunfilesSupplier(
      PathFragment runfilesDir, Runfiles runfiles) {
    return new SingleRunfilesSupplier(
        runfilesDir,
        runfiles,
        /*manifest=*/ null,
        /*repoMappingManifest=*/ null,
        /*buildRunfileLinks=*/ false,
        /*runfileLinksEnabled=*/ false);
  }

  public static BuildOptions execOptions(BuildOptions targetOptions, EventHandler handler)
      throws InterruptedException {
    return Iterables.getOnlyElement(
        ExecutionTransitionFactory.createFactory()
            .create(
                AttributeTransitionData.builder()
                    .attributes(FakeAttributeMapper.empty())
                    .executionPlatform(Label.parseCanonicalUnchecked("//platform:exec"))
                    .build())
            .apply(new BuildOptionsView(targetOptions, targetOptions.getFragmentClasses()), handler)
            .values());
  }
}
