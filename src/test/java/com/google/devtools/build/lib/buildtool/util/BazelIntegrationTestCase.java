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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.BuildResultListener;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.List;
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

  @CanIgnoreReturnValue
  protected BazelServer server() {
    if (bazel == null) {
      bazel = getServerBuilder().build();
      events = bazel.events();
    }
    return bazel;
  }

  // --- Command Execution ---

  @CanIgnoreReturnValue
  public CommandResult buildTarget(String... targets) throws Exception {
    return server().build(targets);
  }

  @CanIgnoreReturnValue
  public CommandResult buildTarget(List<String> targets) throws Exception {
    return server().build(targets.toArray(String[]::new));
  }

  public void addOptions(String... options) {
    server().addOptions(options);
  }

  public void addOptions(List<String> options) {
    server().addOptions(options);
  }

  // --- Workspace Operations ---

  protected TestWorkspace workspace() {
    return server().workspace();
  }

  @CanIgnoreReturnValue
  public Path write(String workspaceRelativePath, String... lines) throws IOException {
    return server().workspace().write(workspaceRelativePath, lines);
  }

  // --- State Introspection ---

  protected PackageManager getPackageManager() {
    return server().getPackageManager();
  }

  protected BuildResultListener getBuildResultListener() {
    return server().getBuildResultListener();
  }

  protected ImmutableSet<ConfiguredTarget> getAnalyzedTargets() {
    return server().getAnalyzedTargets();
  }

  protected ImmutableList<String> getLabelsOfAnalyzedTargets() {
    return server().getLabelsOfAnalyzedTargets();
  }

  protected ImmutableSet<ConfiguredTargetKey> getBuiltTargets() {
    return server().getBuiltTargets();
  }

  protected ImmutableList<String> getLabelsOfBuiltTargets() {
    return server().getLabelsOfBuiltTargets();
  }

  protected ImmutableSet<AspectKey> getAnalyzedAspectKeys() {
    return server().getAnalyzedAspectKeys();
  }

  protected ImmutableList<String> getLabelsOfAnalyzedAspects() {
    return server().getLabelsOfAnalyzedAspects();
  }

  protected ImmutableSet<AspectKey> getBuiltAspects() {
    return server().getBuiltAspects();
  }

  protected ImmutableList<String> getLabelsOfBuiltAspects() {
    return server().getLabelsOfBuiltAspects();
  }
}
