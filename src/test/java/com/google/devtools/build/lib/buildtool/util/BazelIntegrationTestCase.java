// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.BuildResultListener;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import org.junit.After;
import org.junit.Before;

/**
 * Convenience base class for Bazel integration tests, powered by an in-process {@link BazelServer}.
 *
 * <p>Provides helper methods for building targets, manipulating workspace files, and inspecting
 * internal server state.
 */
public abstract class BazelIntegrationTestCase {

  protected BazelServer bazel;

  /** Backwards-compatible events apparatus. Available after {@link #setUpServer()}. */
  protected EventCollectionApparatus events;

  /**
   * Returns additional types of events for {@link #events} to collect.
   *
   * <p>{@link EventKind#ERRORS_WARNINGS_AND_INFO} are always collected by default.
   */
  protected Set<EventKind> additionalEventsToCollect() {
    return ImmutableSet.of();
  }

  /**
   * Returns the {@link BazelServer.Builder} used to configure and construct the test server.
   *
   * <p>Subclasses can override this method to add custom modules.
   */
  protected BazelServer.Builder getServerBuilder() {
    return BazelServer.builder();
  }

  @Before
  public void setUpServer() throws Exception {
    server();
  }

  @After
  public void tearDownServer() throws Exception {
    if (bazel != null) {
      try {
        bazel.close();
      } finally {
        bazel = null;
        events = null;
      }
    }
  }

  /** Returns the lazily-initialized {@link BazelServer} instance for this test. */
  @CanIgnoreReturnValue
  protected BazelServer server() {
    if (bazel == null) {
      BazelServer.Builder builder = getServerBuilder();
      Set<EventKind> additionalEvents = additionalEventsToCollect();
      if (!additionalEvents.isEmpty()) {
        builder.withAdditionalEventsToCollect(additionalEvents);
      }
      bazel = builder.build();
      events = bazel.events();
    }
    return bazel;
  }

  // --- Command Execution ---

  /** Runs {@code build} on the specified targets and returns the {@link CommandResult}. */
  @CanIgnoreReturnValue
  public CommandResult buildTarget(String... targets) throws Exception {
    return server().build(targets);
  }

  /** Runs {@code build} on the specified targets and returns the {@link CommandResult}. */
  @CanIgnoreReturnValue
  public CommandResult buildTarget(List<String> targets) throws Exception {
    return server().build(targets.toArray(String[]::new));
  }

  /** Adds persistent command-line options that will be passed to subsequent commands. */
  public void addOptions(String... options) {
    server().addOptions(options);
  }

  /** Adds persistent command-line options that will be passed to subsequent commands. */
  public void addOptions(List<String> options) {
    server().addOptions(options);
  }

  // --- Workspace Operations ---

  /** Returns the {@link TestWorkspace} for manipulating files in the test workspace. */
  protected TestWorkspace workspace() {
    return server().workspace();
  }

  /** Writes lines to a workspace-relative path and returns the written {@link Path}. */
  @CanIgnoreReturnValue
  public Path write(String workspaceRelativePath, String... lines) throws IOException {
    return server().workspace().write(workspaceRelativePath, lines);
  }

  // --- State Introspection ---

  /** Returns the {@link PackageManager} from the current {@link SkyframeExecutor}. */
  protected PackageManager getPackageManager() {
    return server().getPackageManager();
  }

  /** Returns the {@link BuildResultListener} from the most recent build, or {@code null}. */
  protected BuildResultListener getBuildResultListener() {
    return server().getBuildResultListener();
  }

  /**
   * Returns the {@link ConfiguredTarget} for {@code label} using the target configuration from the
   * most recent build, evaluating it in Skyframe if needed.
   */
  protected ConfiguredTarget getConfiguredTarget(String label) throws Exception {
    return server().getConfiguredTarget(label);
  }

  /**
   * Returns all {@link ConfiguredTarget}s currently present in the Skyframe graph, including
   * transitive dependencies.
   */
  protected ImmutableList<ConfiguredTarget> getAllConfiguredTargets() {
    return server().getAllConfiguredTargets();
  }

  /**
   * Returns an already-computed {@link ConfiguredTarget} from the Skyframe graph for {@code target}
   * using the target configuration from the most recent build, asserting that it exists without
   * evaluating new Skyframe nodes.
   */
  @CanIgnoreReturnValue
  protected ConfiguredTarget getExistingConfiguredTarget(String target) throws Exception {
    return server().getExistingConfiguredTarget(target);
  }

  /** Returns the files to build for the given {@link TransitiveInfoCollection}. */
  protected NestedSet<Artifact> getFilesToBuild(TransitiveInfoCollection target) {
    return target.getProvider(FileProvider.class).getFilesToBuild();
  }

  /** Reads the contents of {@code artifact} as a Latin-1 string. */
  protected String readContentAsLatin1String(Artifact artifact) throws IOException {
    return new String(FileSystemUtils.readContentAsLatin1(artifact.getPath()));
  }

  /** Returns the {@link ActionGraph} from the current {@link SkyframeExecutor}. */
  protected ActionGraph getActionGraph() {
    return server().getActionGraph();
  }

  /** Returns the top-level targets analyzed in the most recent build. */
  protected ImmutableSet<ConfiguredTarget> getAnalyzedTargets() {
    return server().getAnalyzedTargets();
  }

  /** Returns the label strings of the top-level targets analyzed in the most recent build. */
  protected ImmutableList<String> getLabelsOfAnalyzedTargets() {
    return server().getLabelsOfAnalyzedTargets();
  }

  /** Returns the top-level targets built in the most recent build. */
  protected ImmutableSet<ConfiguredTargetKey> getBuiltTargets() {
    return server().getBuiltTargets();
  }

  /** Returns the label strings of the top-level targets built in the most recent build. */
  protected ImmutableList<String> getLabelsOfBuiltTargets() {
    return server().getLabelsOfBuiltTargets();
  }

  /** Returns the aspect keys analyzed in the most recent build. */
  protected ImmutableSet<AspectKey> getAnalyzedAspectKeys() {
    return server().getAnalyzedAspectKeys();
  }

  /** Returns the label strings of the aspects analyzed in the most recent build. */
  protected ImmutableList<String> getLabelsOfAnalyzedAspects() {
    return server().getLabelsOfAnalyzedAspects();
  }

  /** Returns the aspect keys built in the most recent build. */
  protected ImmutableSet<AspectKey> getBuiltAspects() {
    return server().getBuiltAspects();
  }

  /** Returns the label strings of the aspects built in the most recent build. */
  protected ImmutableList<String> getLabelsOfBuiltAspects() {
    return server().getLabelsOfBuiltAspects();
  }
}
