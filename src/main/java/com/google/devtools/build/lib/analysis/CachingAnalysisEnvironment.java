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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoCollection;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.WorkspaceStatusValue;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import javax.annotation.Nullable;

/**
 * The implementation of AnalysisEnvironment used for analysis. It tracks metadata for each
 * configured target, such as the errors and warnings emitted by that target. It is intended that
 * a separate instance is used for each configured target, so that these don't mix up.
 */
public class CachingAnalysisEnvironment implements AnalysisEnvironment {
  private final ArtifactFactory artifactFactory;

  private final ArtifactOwner owner;
  /**
   * If this is the system analysis environment, then errors and warnings are directly reported
   * to the global reporter, rather than stored, i.e., we don't track here whether there are any
   * errors.
   */
  private final boolean isSystemEnv;
  private final boolean extendedSanityChecks;

  /**
   * If false, no actions will be registered, they'll all be just dropped.
   *
   * <p>Usually, an analysis environment should register all actions. However, in some scenarios we
   * analyze some targets twice, but the first one only serves the purpose of collecting information
   * for the second analysis. In this case we don't register actions created by the first pass in
   * order to avoid action conflicts.
   */
  private final boolean allowRegisteringActions;

  private boolean enabled = true;
  private MiddlemanFactory middlemanFactory;
  private EventHandler errorEventListener;
  private SkyFunction.Environment skyframeEnv;
  private Map<Artifact, String> artifacts;
  private final BinTools binTools;

  /**
   * The list of actions registered by the configured target this analysis environment is
   * responsible for. May get cleared out at the end of the analysis of said target.
   */
  final List<Action> actions = new ArrayList<>();

  public CachingAnalysisEnvironment(ArtifactFactory artifactFactory,
      ArtifactOwner owner, boolean isSystemEnv, boolean extendedSanityChecks,
      EventHandler errorEventListener, SkyFunction.Environment env, boolean allowRegisteringActions,
      BinTools binTools) {
    this.artifactFactory = artifactFactory;
    this.owner = Preconditions.checkNotNull(owner);
    this.isSystemEnv = isSystemEnv;
    this.extendedSanityChecks = extendedSanityChecks;
    this.errorEventListener = errorEventListener;
    this.skyframeEnv = env;
    this.allowRegisteringActions = allowRegisteringActions;
    this.binTools = Preconditions.checkNotNull(binTools);
    middlemanFactory = new MiddlemanFactory(artifactFactory, this);
    artifacts = new HashMap<>();
  }

  public void disable(Target target) {
    if (!hasErrors() && allowRegisteringActions) {
      verifyGeneratedArtifactHaveActions(target);
    }
    artifacts = null;
    middlemanFactory = null;
    enabled = false;
    errorEventListener = null;
    skyframeEnv = null;
  }

  private static StringBuilder shortDescription(Action action) {
    if (action == null) {
      return new StringBuilder("null Action");
    }
    return new StringBuilder()
      .append(action.getClass().getName())
      .append(' ')
      .append(action.getMnemonic());
  }

  /**
   * Sanity checks that all generated artifacts have a generating action.
   * @param target for error reporting
   */
  public void verifyGeneratedArtifactHaveActions(Target target) {
    Collection<String> orphanArtifacts = getOrphanArtifactMap().values();
    List<String> checkedActions = null;
    if (!orphanArtifacts.isEmpty()) {
      checkedActions = Lists.newArrayListWithCapacity(actions.size());
      for (Action action : actions) {
        StringBuilder sb = shortDescription(action);
        for (Artifact o : action.getOutputs()) {
          sb.append("\n    ");
          sb.append(o.getExecPathString());
        }
        checkedActions.add(sb.toString());
      }
      throw new IllegalStateException(
          String.format(
              "%s %s : These artifacts do not have a generating action:\n%s\n"
              + "These actions were checked:\n%s\n",
              target.getTargetKind(), target.getLabel(),
              Joiner.on('\n').join(orphanArtifacts), Joiner.on('\n').join(checkedActions)));
    }
  }

  @Override
  public ImmutableSet<Artifact> getOrphanArtifacts() {
    if (!allowRegisteringActions) {
      return ImmutableSet.<Artifact>of();
    }
    return ImmutableSet.copyOf(getOrphanArtifactMap().keySet());
  }

  private Map<Artifact, String> getOrphanArtifactMap() {
    // Construct this set to avoid poor performance under large --runs_per_test.
    Set<Artifact> artifactsWithActions = new HashSet<>();
    for (Action action : actions) {
      // Don't bother checking that every Artifact only appears once; that test is performed
      // elsewhere (see #testNonUniqueOutputs in ActionListenerIntegrationTest).
      artifactsWithActions.addAll(action.getOutputs());
    }
    // The order of the artifacts.entrySet iteration is unspecified - we use a TreeMap here to
    // guarantee that the return value of this method is deterministic.
    Map<Artifact, String> orphanArtifacts = new TreeMap<>(Artifact.EXEC_PATH_COMPARATOR);
    for (Map.Entry<Artifact, String> entry : artifacts.entrySet()) {
      Artifact a = entry.getKey();
      if (!a.isSourceArtifact() && !artifactsWithActions.contains(a)) {
        orphanArtifacts.put(a, String.format("%s\n%s",
            a.getExecPathString(),  // uncovered artifact
            entry.getValue()));  // origin of creation
      }
    }
    return orphanArtifacts;
  }

  @Override
  public EventHandler getEventHandler() {
    return errorEventListener;
  }

  @Override
  public boolean hasErrors() {
    // The system analysis environment never has errors.
    if (isSystemEnv) {
      return false;
    }
    Preconditions.checkState(enabled);
    return ((StoredEventHandler) errorEventListener).hasErrors();
  }

  @Override
  public MiddlemanFactory getMiddlemanFactory() {
    Preconditions.checkState(enabled);
    return middlemanFactory;
  }

  /**
   * Keeps track of artifacts. We check that all of them have an owner when the environment is
   * sealed (disable()). For performance reasons we only track the originating stacktrace when
   * running with --experimental_extended_sanity_checks.
   */
  private Artifact trackArtifactAndOrigin(Artifact a, @Nullable Throwable e) {
    if ((e != null) && !artifacts.containsKey(a)) {
      StringWriter sw = new StringWriter();
      e.printStackTrace(new PrintWriter(sw));
      artifacts.put(a, sw.toString());
    } else {
      artifacts.put(a, "No origin, run with --experimental_extended_sanity_checks");
    }
    return a;
  }

  @Override
  public Artifact getDerivedArtifact(PathFragment rootRelativePath, Root root) {
    Preconditions.checkState(enabled);
    return trackArtifactAndOrigin(
        artifactFactory.getDerivedArtifact(rootRelativePath, root, getOwner()),
        extendedSanityChecks ? new Throwable() : null);
  }

  @Override
  public Artifact getTreeArtifact(PathFragment rootRelativePath, Root root) {
    Preconditions.checkState(enabled);
    return trackArtifactAndOrigin(
        artifactFactory.getTreeArtifact(rootRelativePath, root, getOwner()),
        extendedSanityChecks ? new Throwable() : null);
  }

  @Override
  public Artifact getFilesetArtifact(PathFragment rootRelativePath, Root root) {
    Preconditions.checkState(enabled);
    return trackArtifactAndOrigin(
        artifactFactory.getFilesetArtifact(rootRelativePath, root, getOwner()),
        extendedSanityChecks ? new Throwable() : null);
  }

  @Override
  public Artifact getConstantMetadataArtifact(PathFragment rootRelativePath, Root root) {
    return artifactFactory.getConstantMetadataArtifact(rootRelativePath, root, getOwner());
  }

  @Override
  public Artifact getEmbeddedToolArtifact(String embeddedPath) {
    Preconditions.checkState(enabled);
    return binTools.getEmbeddedArtifact(embeddedPath, artifactFactory);
  }

  @Override
  public void registerAction(Action... actions) {
    Preconditions.checkState(enabled);
    if (allowRegisteringActions) {
      Collections.addAll(this.actions, actions);
    }
  }

  @Override
  public Action getLocalGeneratingAction(Artifact artifact) {
    Preconditions.checkState(allowRegisteringActions);
    for (Action action : actions) {
      if (action.getOutputs().contains(artifact)) {
        return action;
      }
    }
    return null;
  }

  @Override
  public Collection<Action> getRegisteredActions() {
    return Collections.unmodifiableCollection(actions);
  }

  @Override
  public SkyFunction.Environment getSkyframeEnv() {
    return skyframeEnv;
  }

  @Override
  public Artifact getStableWorkspaceStatusArtifact() {
    return ((WorkspaceStatusValue) skyframeEnv.getValue(WorkspaceStatusValue.SKY_KEY))
            .getStableArtifact();
  }

  @Override
  public Artifact getVolatileWorkspaceStatusArtifact() {
    return ((WorkspaceStatusValue) skyframeEnv.getValue(WorkspaceStatusValue.SKY_KEY))
            .getVolatileArtifact();
  }

  // See SkyframeBuildView#getWorkspaceStatusValues for the code that this method is attempting to
  // verify.
  private NullPointerException collectDebugInfoAndCrash(
      BuildInfoKey key, BuildConfiguration config) {
    String debugInfo = key + " " + config;
    Preconditions.checkState(skyframeEnv.valuesMissing(), debugInfo);
    Map<BuildInfoKey, BuildInfoFactory> buildInfoFactories = Preconditions.checkNotNull(
        PrecomputedValue.BUILD_INFO_FACTORIES.get(skyframeEnv), debugInfo);
    BuildInfoFactory buildInfoFactory =
        Preconditions.checkNotNull(buildInfoFactories.get(key), debugInfo);
    Preconditions.checkState(buildInfoFactory.isEnabled(config), debugInfo);
    throw new NullPointerException("BuildInfoCollectionValue shouldn't have been null");
  }

  @Override
  public ImmutableList<Artifact> getBuildInfo(RuleContext ruleContext, BuildInfoKey key) {
    boolean stamp = AnalysisUtils.isStampingEnabled(ruleContext);
    BuildInfoCollectionValue collectionValue =
        (BuildInfoCollectionValue) skyframeEnv.getValue(BuildInfoCollectionValue.key(
            new BuildInfoCollectionValue.BuildInfoKeyAndConfig(
                key, ruleContext.getConfiguration())));
    if (collectionValue == null) {
      throw collectDebugInfoAndCrash(key, ruleContext.getConfiguration());
    }
    BuildInfoCollection collection = collectionValue.getCollection();
   return stamp ? collection.getStampedBuildInfo() : collection.getRedactedBuildInfo();
  }

  @Override
  public ArtifactOwner getOwner() {
    return owner;
  }
}
