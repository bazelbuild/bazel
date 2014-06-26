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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;
import com.google.devtools.build.skyframe.NodeOrException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A builder for {@link ActionExecutionNode}s.
 */
public class ActionExecutionNodeBuilder implements NodeBuilder {

  private static final Predicate<Artifact> IS_SOURCE_ARTIFACT = new Predicate<Artifact>() {
      @Override
      public boolean apply(Artifact input) {
        return input.isSourceArtifact();
      }
    };

  private final SkyframeActionExecutor skyframeActionExecutor;
  private final TimestampGranularityMonitor tsgm;

  public ActionExecutionNodeBuilder(SkyframeActionExecutor skyframeActionExecutor,
      TimestampGranularityMonitor tsgm) {
    this.skyframeActionExecutor = skyframeActionExecutor;
    this.tsgm = tsgm;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env) throws ActionExecutionNodeBuilderException,
      InterruptedException {
    Action action = (Action) nodeKey.getNodeName();
    Map<Artifact, FileArtifactNode> inputArtifactData = null;
    Map<Artifact, Collection<Artifact>> expandedMiddlemen = null;
    try {
      Pair<Map<Artifact, FileArtifactNode>, Map<Artifact, Collection<Artifact>>> checkedInputs =
          checkInputs(env, action); // Declare deps on known inputs to action.

      inputArtifactData = checkedInputs.first;
      expandedMiddlemen = checkedInputs.second;
    } catch (ActionExecutionException e) {
      skyframeActionExecutor.postActionNotExecutedEvents(action, e.getRootCauses());
      throw new ActionExecutionNodeBuilderException(nodeKey, e);
    }
    // TODO(bazel-team): Non-volatile NotifyOnActionCacheHit actions perform worse in Skyframe than
    // legacy when they are not at the top of the action graph. In legacy, they are stored
    // separately, so notifying non-dirty actions is cheap. In Skyframe, they depend on the
    // BUILD_ID, forcing invalidation of upward transitive closure on each build.
    if (action.isVolatile() || action instanceof NotifyOnActionCacheHit) {
      // Volatile build actions may need to execute even if none of their known inputs have changed.
      // Depending on the buildID ensure that these actions have a chance to execute.
      BuildVariableNode.BUILD_ID.get(env);
    }
    if (env.depsMissing()) {
      return null;
    }
    FileAndMetadataCache cache = new FileAndMetadataCache(inputArtifactData, expandedMiddlemen,
        skyframeActionExecutor.getExecRoot(), tsgm);
    try {
      skyframeActionExecutor.executeAction(action, cache);
    } catch (ActionExecutionException e) {
      skyframeActionExecutor.reportActionExecutionFailure(action);
      // In this case we do not report the error to the action reporter because we have already
      // done it in SkyframeExecutor.reportErrorIfNotAbortingMode() method. That method prints
      // the error in the top-level reporter and also dumps the recorded StdErr for the action.
      // Label can be null in the case of, e.g., the SystemActionOwner (for build-info.txt).
      skyframeActionExecutor.postActionNotExecutedEvents(action, e.getRootCauses());
      throw new ActionExecutionNodeBuilderException(nodeKey,
          new AlreadyReportedActionExecutionException(e));
    } finally {
      declareAdditionalDependencies(env, action);
    }
    if (env.depsMissing()) {
      return null;
    }

    return new ActionExecutionNode(cache.getOutputData(), cache.getAdditionalOutputData());
  }

  private static Collection<NodeKey> toKeys(Iterable<Artifact> inputs,
      Iterable<Artifact> mandatoryInputs) {
    Set<Artifact> mandatory = Sets.newHashSet(mandatoryInputs);
    Collection<NodeKey> discoveredArtifacts = new ArrayList<>();
    for (Artifact artifact : inputs) {
      discoveredArtifacts.add(ArtifactNode.key(artifact, mandatory.contains(artifact)));
    }
    return discoveredArtifacts;
  }

  /**
   * Declare dependency on all known inputs of action. Throws exception if any are known to be
   * missing. Some inputs may not yet be in the graph, in which case the builder should abort.
   */
  private Pair<Map<Artifact, FileArtifactNode>, Map<Artifact, Collection<Artifact>>> checkInputs(
      Environment env, Action action) throws ActionExecutionException {
    int missingCount = 0;
    ImmutableList.Builder<Label> rootCauses = ImmutableList.builder();
    Map<Artifact, FileArtifactNode> inputArtifactData = new HashMap<>();
    Map<Artifact, Collection<Artifact>> expandedMiddlemen = new HashMap<>();

    ActionExecutionException firstActionExecutionException = null;
    for (Map.Entry<NodeKey, NodeOrException<Exception>> depsEntry :
      env.getDepsOrThrow(toKeys(action.getInputs(), action.getMandatoryInputs()), Exception.class)
          .entrySet()) {
      Artifact input = ArtifactNode.artifact(depsEntry.getKey());
      try {
        ArtifactNode node = (ArtifactNode) depsEntry.getValue().get();
        if (node instanceof AggregatingArtifactNode) {
          AggregatingArtifactNode aggregatingNode = (AggregatingArtifactNode) node;
          Set<Artifact> expansion = new HashSet<>();
          for (Pair<Artifact, FileArtifactNode> entry : aggregatingNode.getInputs()) {
            inputArtifactData.put(entry.first, entry.second);
            expansion.add(entry.first);
          }
          // We have to cache the "digest" of the aggregating node itself, because the action cache
          // checker may want it.
          inputArtifactData.put(input, aggregatingNode.getSelfData());
          expandedMiddlemen.put(input, ImmutableList.copyOf(expansion));
        } else if (node instanceof FileArtifactNode) {
          // TODO(bazel-team): Make sure middleman "virtual" artifact data is properly processed.
          inputArtifactData.put(input, (FileArtifactNode) node);
        }
      } catch (MissingInputFileException e) {
        missingCount++;
        if (input.getOwner() != null) {
          rootCauses.add(input.getOwner());
        }
      } catch (ActionExecutionException e) {
        if (firstActionExecutionException == null) {
          firstActionExecutionException = e;
        }
        skyframeActionExecutor.postActionNotExecutedEvents(action, e.getRootCauses());
      } catch (Exception e) {
        // can't get here
        throw new IllegalStateException(e); 
      }
    }
    // We need to rethrow first exception because it can contain useful error message
    if (firstActionExecutionException != null) {
      throw firstActionExecutionException;
    }

    if (missingCount > 0) {
      throw new ActionExecutionException(missingCount + " input file(s) do not exist", action,
          rootCauses.build(), /*catastrophe=*/false);
    }
    return Pair.<Map<Artifact, FileArtifactNode>, Map<Artifact, Collection<Artifact>>>of(
        ImmutableMap.copyOf(inputArtifactData), ImmutableMap.copyOf(expandedMiddlemen));
  }

  private static void declareAdditionalDependencies(Environment env, Action action) {
    if (action.discoversInputs()) {
      env.getDeps(toKeys(Iterables.filter(action.getInputs(), IS_SOURCE_ARTIFACT),
          action.getMandatoryInputs()));
    }
  }

  /**
   * All info/warning messages associated with actions should be always displayed.
   */
  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ActionExecutionNodeBuilder#build}.
   */
  private static final class ActionExecutionNodeBuilderException extends NodeBuilderException {
    public ActionExecutionNodeBuilderException(NodeKey key, ActionExecutionException e) {
      super(key, e);
    }
  }
}
