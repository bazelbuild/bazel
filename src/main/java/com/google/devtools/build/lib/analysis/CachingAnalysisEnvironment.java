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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoCollection;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionValue;
import com.google.devtools.build.lib.skyframe.StarlarkBuiltinsValue;
import com.google.devtools.build.lib.skyframe.WorkspaceStatusValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * The implementation of AnalysisEnvironment used for analysis. It tracks metadata for each
 * configured target, such as the errors and warnings emitted by that target. It is intended that
 * a separate instance is used for each configured target, so that these don't mix up.
 */
public class CachingAnalysisEnvironment implements AnalysisEnvironment {
  private final ArtifactFactory artifactFactory;

  private final ActionLookupKey owner;
  /**
   * If this is the system analysis environment, then errors and warnings are directly reported
   * to the global reporter, rather than stored, i.e., we don't track here whether there are any
   * errors.
   */
  private final boolean isSystemEnv;
  private final boolean extendedSanityChecks;
  private final boolean allowAnalysisFailures;

  private final ActionKeyContext actionKeyContext;

  private boolean enabled = true;
  private MiddlemanFactory middlemanFactory;
  private ExtendedEventHandler errorEventListener;
  private SkyFunction.Environment skyframeEnv;
  // TODO(bazel-team): Should this be nulled out by disable()? Alternatively, does disable() even
  // need to exist?
  private final StarlarkBuiltinsValue starlarkBuiltinsValue;
  /**
   * Map of artifacts to either themselves or to {@code Pair<Artifact, String>} if
   * --experimental_extended_sanity_checks is enabled. In the latter case, the string will contain
   * the stack trace of where the artifact was created. In the former case, we'll construct a
   * generic message in case of error.
   *
   * <p>The artifact is stored so that we can deduplicate artifacts created multiple times.
   */
  private Map<Artifact, Object> artifacts;

  /**
   * The list of actions registered by the configured target this analysis environment is
   * responsible for. May get cleared out at the end of the analysis of said target.
   */
  final List<ActionAnalysisMetadata> actions = new ArrayList<>();

  public CachingAnalysisEnvironment(
      ArtifactFactory artifactFactory,
      ActionKeyContext actionKeyContext,
      ActionLookupKey owner,
      boolean isSystemEnv,
      boolean extendedSanityChecks,
      boolean allowAnalysisFailures,
      ExtendedEventHandler errorEventListener,
      SkyFunction.Environment env,
      StarlarkBuiltinsValue starlarkBuiltinsValue) {
    this.artifactFactory = artifactFactory;
    this.actionKeyContext = actionKeyContext;
    this.owner = Preconditions.checkNotNull(owner);
    this.isSystemEnv = isSystemEnv;
    this.extendedSanityChecks = extendedSanityChecks;
    this.allowAnalysisFailures = allowAnalysisFailures;
    this.errorEventListener = errorEventListener;
    this.skyframeEnv = env;
    this.starlarkBuiltinsValue = starlarkBuiltinsValue;
    middlemanFactory = new MiddlemanFactory(artifactFactory, this);
    artifacts = new HashMap<>();
  }

  public void disable(Target target) {
    if (!hasErrors() && !allowAnalysisFailures) {
      verifyGeneratedArtifactHaveActions(target);
    }
    artifacts = null;
    middlemanFactory = null;
    enabled = false;
    errorEventListener = null;
    skyframeEnv = null;
  }

  private static StringBuilder shortDescription(ActionAnalysisMetadata action) {
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
      for (ActionAnalysisMetadata action : actions) {
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
    return ImmutableSet.copyOf(getOrphanArtifactMap().keySet());
  }

  @Override
  public ImmutableSet<Artifact> getTreeArtifactsConflictingWithFiles() {
    boolean hasTreeArtifacts = false;
    for (Artifact artifact : artifacts.keySet()) {
      if (artifact.isTreeArtifact()) {
        hasTreeArtifacts = true;
        break;
      }
    }
    if (!hasTreeArtifacts) {
      return ImmutableSet.of();
    }

    HashSet<PathFragment> collect = new HashSet<>();
    for (Artifact artifact : artifacts.keySet()) {
      if (!artifact.isSourceArtifact() && !artifact.isTreeArtifact()) {
        collect.add(artifact.getExecPath());
      }
    }

    ImmutableSet.Builder<Artifact> sameExecPathTreeArtifacts = ImmutableSet.builder();
    for (Artifact artifact : artifacts.keySet()) {
      if (artifact.isTreeArtifact() && collect.contains(artifact.getExecPath())) {
        sameExecPathTreeArtifacts.add(artifact);
      }
    }

    return sameExecPathTreeArtifacts.build();
  }

  private Map<Artifact, String> getOrphanArtifactMap() {
    // Construct this set to avoid poor performance under large --runs_per_test.
    Set<Artifact> artifactsWithActions = new HashSet<>();
    for (ActionAnalysisMetadata action : actions) {
      // Don't bother checking that every Artifact only appears once; that test is performed
      // elsewhere (see #testNonUniqueOutputs in ActionListenerIntegrationTest).
      artifactsWithActions.addAll(action.getOutputs());
    }
    // The order of the artifacts.entrySet iteration is unspecified - we use a TreeMap here to
    // guarantee that the return value of this method is deterministic.
    Map<Artifact, String> orphanArtifacts = new TreeMap<>(Artifact.EXEC_PATH_COMPARATOR);
    for (Map.Entry<Artifact, Object> entry : artifacts.entrySet()) {
      Artifact a = entry.getKey();
      if (!a.isSourceArtifact() && !artifactsWithActions.contains(a)) {
        Object value = entry.getValue();
        if (value instanceof Artifact) {
          value = "No origin, run with --experimental_extended_sanity_checks";
        } else {
          value = ((Pair<?, ?>) value).second;
        }
        orphanArtifacts.put(
            a,
            String.format(
                "%s\n%s",
                a.getExecPathString(), // uncovered artifact
                value)); // origin of creation
      }
    }
    return orphanArtifacts;
  }

  @Override
  public ExtendedEventHandler getEventHandler() {
    return errorEventListener;
  }

  @Override
  public ActionKeyContext getActionKeyContext() {
    return actionKeyContext;
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
  @SuppressWarnings("unchecked") // Cast of artifacts map's value to Pair.
  private Artifact.DerivedArtifact dedupAndTrackArtifactAndOrigin(
      Artifact.DerivedArtifact a, @Nullable Throwable e) {
    if (artifacts.containsKey(a)) {
      Object value = artifacts.get(a);
      if (e == null) {
        return (Artifact.DerivedArtifact) value;
      } else {
        return ((Pair<Artifact.DerivedArtifact, String>) value).first;
      }
    }
    if ((e != null)) {
      StringWriter sw = new StringWriter();
      e.printStackTrace(new PrintWriter(sw));
      artifacts.put(a, Pair.of(a, sw.toString()));
    } else {
      artifacts.put(a, a);
    }
    return a;
  }

  @Override
  public Artifact.DerivedArtifact getDerivedArtifact(
      PathFragment rootRelativePath, ArtifactRoot root) {
    return getDerivedArtifact(rootRelativePath, root, /*contentBasedPath=*/ false);
  }

  @Override
  public Artifact.DerivedArtifact getDerivedArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, boolean contentBasedPath) {
    Preconditions.checkState(enabled);
    return dedupAndTrackArtifactAndOrigin(
        artifactFactory.getDerivedArtifact(rootRelativePath, root, getOwner(), contentBasedPath),
        extendedSanityChecks ? new Throwable() : null);
  }

  @Override
  public SpecialArtifact getTreeArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
    Preconditions.checkState(enabled);
    return (SpecialArtifact)
        dedupAndTrackArtifactAndOrigin(
            artifactFactory.getTreeArtifact(rootRelativePath, root, getOwner()),
            extendedSanityChecks ? new Throwable() : null);
  }

  @Override
  public SpecialArtifact getSymlinkArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
    Preconditions.checkState(enabled);
    return (SpecialArtifact)
        dedupAndTrackArtifactAndOrigin(
            artifactFactory.getSymlinkArtifact(rootRelativePath, root, getOwner()),
            extendedSanityChecks ? new Throwable() : null);
  }

  @Override
  public Artifact getSourceArtifactForNinjaBuild(PathFragment execPath, Root root) {
    return artifactFactory.getSourceArtifact(execPath, root, owner);
  }

  @Override
  public Artifact.DerivedArtifact getFilesetArtifact(
      PathFragment rootRelativePath, ArtifactRoot root) {
    Preconditions.checkState(enabled);
    return dedupAndTrackArtifactAndOrigin(
        artifactFactory.getFilesetArtifact(rootRelativePath, root, getOwner()),
        extendedSanityChecks ? new Throwable() : null);
  }

  @Override
  public Artifact getConstantMetadataArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
    return artifactFactory.getConstantMetadataArtifact(rootRelativePath, root, getOwner());
  }

  @Override
  public void registerAction(ActionAnalysisMetadata action) {
    Preconditions.checkState(enabled);
    this.actions.add(Preconditions.checkNotNull(action, owner));
  }

  @Override
  public ActionAnalysisMetadata getLocalGeneratingAction(Artifact artifact) {
    for (ActionAnalysisMetadata action : actions) {
      if (action.getOutputs().contains(artifact)) {
        return action;
      }
    }
    return null;
  }

  @Override
  public ImmutableList<ActionAnalysisMetadata> getRegisteredActions() {
    return ImmutableList.copyOf(actions);
  }

  @Override
  public SkyFunction.Environment getSkyframeEnv() {
    return skyframeEnv;
  }

  @Override
  public StarlarkSemantics getStarlarkSemantics() throws InterruptedException {
    return starlarkBuiltinsValue.starlarkSemantics;
  }

  @Override
  public ImmutableMap<String, Object> getStarlarkDefinedBuiltins() throws InterruptedException {
    return starlarkBuiltinsValue.exportedToJava;
  }

  @Override
  public Artifact getStableWorkspaceStatusArtifact() throws InterruptedException {
    return getWorkspaceStatusValue().getStableArtifact();
  }

  @Override
  public Artifact getVolatileWorkspaceStatusArtifact() throws InterruptedException {
    return getWorkspaceStatusValue().getVolatileArtifact();
  }

  private WorkspaceStatusValue getWorkspaceStatusValue() throws InterruptedException {
    WorkspaceStatusValue workspaceStatusValue =
        ((WorkspaceStatusValue) skyframeEnv.getValue(WorkspaceStatusValue.BUILD_INFO_KEY));
    if (workspaceStatusValue == null) {
      throw new MissingDepException("Restart due to missing build info");
    }
    return workspaceStatusValue;
  }

  @Override
  public ImmutableList<Artifact> getBuildInfo(
      boolean stamp, BuildInfoKey key, BuildConfiguration config) throws InterruptedException {
    BuildInfoCollectionValue collectionValue =
        (BuildInfoCollectionValue) skyframeEnv.getValue(BuildInfoCollectionValue.key(key, config));
    if (collectionValue == null) {
      throw new MissingDepException(
          String.format("Restart due to missing BuildInfoCollectionValue (%s %s)", key, config));
    }
    BuildInfoCollection collection = collectionValue.getCollection();
    return stamp ? collection.getStampedBuildInfo() : collection.getRedactedBuildInfo();
  }

  @Override
  public ActionLookupKey getOwner() {
    return owner;
  }

  /** Thrown in case of a missing build info key. */
  // TODO(ulfjack): It would be better for this to be a checked exception, which requires updating
  // all callers to pass the exception through.
  public static class MissingDepException extends RuntimeException {
    MissingDepException(String msg) {
      super(msg);
    }
  }
}
