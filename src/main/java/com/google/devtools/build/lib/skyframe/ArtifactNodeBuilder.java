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

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Action.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.MissingArtifactEvent;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.skyframe.ActionLookupNode.ActionLookupKey;
import com.google.devtools.build.lib.skyframe.ArtifactNode.OwnedArtifact;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A builder for {@link ArtifactNode}s.
 */
class ArtifactNodeBuilder implements NodeBuilder {

  private final AtomicReference<EventBus> eventBus;
  private final Predicate<PathFragment> allowedMissingInputs;

  ArtifactNodeBuilder(AtomicReference<EventBus> eventBus,
      Predicate<PathFragment> allowedMissingInputs) {
    this.eventBus = eventBus;
    this.allowedMissingInputs = allowedMissingInputs;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env) throws ArtifactNodeBuilderException {
    OwnedArtifact ownedArtifact = (OwnedArtifact) nodeKey.getNodeName();
    Artifact artifact = ownedArtifact.getArtifact();
    if (artifact.isSourceArtifact()) {
      try {
        return createSourceNode(artifact, ownedArtifact.isMandatory(), env);
      } catch (MissingInputFileException e) {
        if (eventBus.get() != null) {
          eventBus.get().post(new MissingArtifactEvent(artifact.getOwner()));
        }
        throw new ArtifactNodeBuilderException(nodeKey, e);
      }
    }

    Action action = extractActionFromArtifact(artifact, env);
    if (action == null) {
      return null;
    }

    ActionExecutionNode actionNode =
        (ActionExecutionNode) env.getDep(ActionExecutionNode.key(action));
    if (actionNode == null) {
      return null;
    }

    if (!isAggregatingNode(action)) {
      try {
        return createSimpleNode(artifact, actionNode);
      } catch (IOException e) {
        ActionExecutionException ex = new ActionExecutionException(e, action,
            /*catastrophe=*/false);
        env.getListener().error(ex.getLocation(), ex.getMessage());
        throw new ArtifactNodeBuilderException(nodeKey, ex);
      }
    } else {
      return createAggregatingNode(artifact, action, actionNode.getArtifactNode(artifact), env);
    }
  }

  private ArtifactNode createSourceNode(Artifact artifact, boolean mandatory, Environment env)
      throws MissingInputFileException {
    NodeKey fileNodeKey = FileNode.key(artifact);
    FileNode fileNode;
    try {
      fileNode = (FileNode) env.getDepOrThrow(fileNodeKey, Exception.class);
    } catch (IOException | InconsistentFilesystemException | FileSymlinkCycleException e) {
      return missingInputFile(artifact, mandatory, e, env.getListener());
    } catch (Exception e) {
      // Can't get here.
      throw new IllegalStateException(e);
    }
    if (fileNode == null) {
      return null;
    }
    if (!fileNode.exists()) {
      if (allowedMissingInputs.apply(((RootedPath) fileNodeKey.getNodeName()).getRelativePath())) {
        return ArtifactNode.newEmptyNode();
      } else {
        return missingInputFile(artifact, mandatory, null, env.getListener());
      }
    }
    try {
      return FileArtifactNode.create(artifact, fileNode);
    } catch (IOException e) {
      return missingInputFile(artifact, mandatory, e, env.getListener());
    }
  }

  private static ArtifactNode missingInputFile(Artifact artifact, boolean mandatory,
                                               Exception failure,
                                               ErrorEventListener reporter)
      throws MissingInputFileException {
    if (!mandatory) {
      return ArtifactNode.newMissingNode();
    }
    String extraMsg = (failure == null) ? "" : (":" + failure.getMessage());
    MissingInputFileException ex = new MissingInputFileException(
        constructErrorMessage(artifact) + extraMsg, null);
    reporter.error(ex.getLocation(), ex.getMessage());
    throw ex;
  }

  // Non-aggregating artifact -- should contain at most one piece of artifact data.
  // data may be null if and only if artifact is a middleman artifact.
  private ArtifactNode createSimpleNode(Artifact artifact, ActionExecutionNode actionNode)
      throws IOException {
    ArtifactNode node = actionNode.getArtifactNode(artifact);
    if (node != null) {
      return node;
    }
    // Middleman artifacts have no corresponding files, so their ArtifactNodes should have already
    // been constructed during execution of the action.
    Preconditions.checkState(!artifact.isMiddlemanArtifact(), artifact);
    FileNode data = Preconditions.checkNotNull(actionNode.getData(artifact),
        "%s %s", artifact, actionNode);
    Preconditions.checkNotNull(data.getDigest(),
          "Digest should already have been calculated for %s (%s)", artifact, data);
    return FileArtifactNode.create(artifact, data);
  }

  private AggregatingArtifactNode createAggregatingNode(Artifact artifact, Action action,
      FileArtifactNode node, NodeBuilder.Environment env) {
    // This artifact aggregates other artifacts. Keep track of them so callers can find them.
    ImmutableList.Builder<Pair<Artifact, FileArtifactNode>> inputs = ImmutableList.builder();
    for (Map.Entry<NodeKey, Node> entry :
        env.getDeps(ArtifactNode.mandatoryKeys(action.getInputs())).entrySet()) {
      Artifact input = ArtifactNode.artifact(entry.getKey());
      ArtifactNode inputNode = (ArtifactNode) entry.getValue();
      Preconditions.checkNotNull(inputNode, "%s has null dep %s", artifact, input);
      if (!(inputNode instanceof FileArtifactNode)) {
        // We do not recurse in aggregating middleman artifacts.
        Preconditions.checkState(!(inputNode instanceof AggregatingArtifactNode),
            "%s %s %s", artifact, action, inputNode);
        continue;
      }
      inputs.add(Pair.of(input, (FileArtifactNode) inputNode));
    }
    return new AggregatingArtifactNode(inputs.build(), node);
  }

  /**
   * Returns whether this node needs to contain the data of all its inputs. Currently only tests to
   * see if the action is an aggregating middleman action. However, may include runfiles middleman
   * actions and Fileset artifacts in the future.
   */
  private static boolean isAggregatingNode(Action action) {
    return action.getActionType() == MiddlemanType.AGGREGATING_MIDDLEMAN;
  }

  @Override
  public String extractTag(NodeKey nodeKey) {
    return Label.print(((OwnedArtifact) nodeKey.getNodeName()).getArtifact().getOwner());
  }

  private Action extractActionFromArtifact(Artifact artifact, NodeBuilder.Environment env) {
    ArtifactOwner artifactOwner = artifact.getArtifactOwner();

    Preconditions.checkState(artifactOwner instanceof ActionLookupKey, "", artifact, artifactOwner);
    NodeKey actionLookupKey = ActionLookupNode.key((ActionLookupKey) artifactOwner);
    ActionLookupNode node = (ActionLookupNode) env.getDep(actionLookupKey);
    if (node == null) {
      // TargetCompletionActionNodes are created on demand. All others should already exist --
      // ConfiguredTargetNodes were created during the analysis phase, and BuildInfo*Nodes were
      // created during the first analysis of a configured target.
      Preconditions.checkState(artifactOwner instanceof TargetCompletionKey,
          "Owner %s of %s not in graph %s", artifactOwner, artifact, actionLookupKey);
      return null;
    }
    return Preconditions.checkNotNull(node.getGeneratingAction(artifact),
          "Node %s does not contain generating action of %s", node, artifact);
  }

  private static final class ArtifactNodeBuilderException extends NodeBuilderException {
    ArtifactNodeBuilderException(NodeKey key, MissingInputFileException e) {
      super(key, e);
    }

    ArtifactNodeBuilderException(NodeKey key, ActionExecutionException e) {
      super(key, e);
    }
  }

  private static String constructErrorMessage(Artifact artifact) {
    if (artifact.getOwner() == null) {
      return String.format("missing input file '%s'", artifact.getPath().getPathString());
    } else {
      return String.format("missing input file '%s'", artifact.getOwner());
    }
  }
}
