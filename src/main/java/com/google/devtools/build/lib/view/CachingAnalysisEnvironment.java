// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.view;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionValue;
import com.google.devtools.build.lib.skyframe.WorkspaceStatusValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoCollection;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.view.config.BinTools;
import com.google.devtools.build.skyframe.SkyFunction;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * The implementation of AnalysisEnvironment used for analysis. It tracks metadata for each
 * configured target, such as the errors and warnings emitted by that target. It is intended that
 * a separate instance is used for each configured target, so that these don't mix up.
 */
public class CachingAnalysisEnvironment implements AnalysisEnvironment {
  private final ArtifactFactory artifactFactory;
  // This is null when we use skyframe to obtain the actions (in skyframe execution), as well as in
  // unit tests.
  @Nullable private final WorkspaceStatusArtifacts workspaceStatusArtifacts;
  private final ImmutableList<OutputFormatter> outputFormatters;

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
  final ArrayList<Action> actions = new ArrayList<>();

  public CachingAnalysisEnvironment(ArtifactFactory artifactFactory,
      ArtifactOwner owner, WorkspaceStatusArtifacts buildInfoHeaders,
      boolean isSystemEnv, boolean extendedSanityChecks, EventHandler errorEventListener,
      SkyFunction.Environment env, boolean allowRegisteringActions,
      ImmutableList<OutputFormatter> outputFormatters, BinTools binTools) {
    this.artifactFactory = artifactFactory;
    this.workspaceStatusArtifacts = buildInfoHeaders;
    this.owner = Preconditions.checkNotNull(owner);
    this.isSystemEnv = isSystemEnv;
    this.extendedSanityChecks = extendedSanityChecks;
    this.errorEventListener = errorEventListener;
    this.skyframeEnv = env;
    this.allowRegisteringActions = allowRegisteringActions;
    this.outputFormatters = outputFormatters;
    this.binTools = binTools;
    middlemanFactory = new MiddlemanFactory(artifactFactory, this);
    artifacts = Maps.newHashMap();
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
    // Construct this set to avoid poor performance under large --runs_per_test.
    Set<Artifact> artifactsWithActions = Sets.newHashSet();
    for (Action action : actions) {
      // Don't bother checking that every Artifact only appears once; that test is performed
      // elsewhere (see #testNonUniqueOutputs in ActionListenerIntegrationTest).
      artifactsWithActions.addAll(action.getOutputs());
    }
    List<String> orphanArtifacts = Lists.newArrayListWithCapacity(artifacts.size());
    List<String> checkedActions = null;
    for (Map.Entry<Artifact, String> entry : artifacts.entrySet()) {
      Artifact a = entry.getKey();
      if (!a.isSourceArtifact() && !artifactsWithActions.contains(a)) {
        orphanArtifacts.add(String.format("%s\n%s",
            a.getExecPathString(),  // uncovered artifact
            entry.getValue()));  // origin of creation
      }
    }
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
              "%s %s : These artifacts miss a generating action:\n%s\n"
              + "These actions we checked:\n%s\n",
              target.getTargetKind(), target.getLabel(),
              Joiner.on('\n').join(orphanArtifacts), Joiner.on('\n').join(checkedActions)));
    }
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

  @Deprecated
  @Override
  public Iterable<OutputFormatter> getOutputFormattersForGenQueryHack() {
    Preconditions.checkState(enabled);
    return outputFormatters;
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
  public Artifact getFilesetArtifact(PathFragment rootRelativePath, Root root) {
    Preconditions.checkState(enabled);
    return trackArtifactAndOrigin(
        artifactFactory.getFilesetArtifact(rootRelativePath, root, getOwner()),
        extendedSanityChecks ? new Throwable() : null);
  }

  @Override
  public Artifact getSpecialMetadataHandlingArtifact(PathFragment rootRelativePath, Root root,
      boolean forceConstantMetadata, boolean forceDigestMetadata) {
    return artifactFactory.getSpecialMetadataHandlingArtifact(rootRelativePath, root,
        getOwner(), forceConstantMetadata, forceDigestMetadata);
  }

  @Override
  public Artifact getEmbeddedToolArtifact(String embeddedPath) {
    Preconditions.checkState(enabled);
    return binTools.getEmbeddedArtifact(embeddedPath, artifactFactory);
  }

  @Override
  public void registerAction(Action action) {
    Preconditions.checkState(enabled);
    if (allowRegisteringActions) {
      actions.add(action);
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
    return workspaceStatusArtifacts == null
        ? ((WorkspaceStatusValue) skyframeEnv.getValue(WorkspaceStatusValue.SKY_KEY))
            .getStableArtifact()
        : workspaceStatusArtifacts.getStableStatus();
  }

  @Override
  public Artifact getVolatileWorkspaceStatusArtifact() {
    return workspaceStatusArtifacts == null
        ? ((WorkspaceStatusValue) skyframeEnv.getValue(WorkspaceStatusValue.SKY_KEY))
            .getVolatileArtifact()
        : workspaceStatusArtifacts.getVolatileStatus();
  }

  @Override
  public ImmutableList<Artifact> getBuildInfo(RuleContext ruleContext, BuildInfoKey key) {
    boolean stamp = AnalysisUtils.isStampingEnabled(ruleContext);
    if (workspaceStatusArtifacts == null) {
      BuildInfoCollection collection =
          ((BuildInfoCollectionValue) skyframeEnv.getValue(BuildInfoCollectionValue.key(
          new BuildInfoCollectionValue.BuildInfoKeyAndConfig(key, ruleContext.getConfiguration()))))
          .getCollection();
      return stamp ? collection.getStampedBuildInfo() : collection.getRedactedBuildInfo();
    }
    return workspaceStatusArtifacts.getBuildInfo(ruleContext.getConfiguration(), key, stamp);
  }

  @Override
  public ArtifactOwner getOwner() {
    return owner;
  }
}
