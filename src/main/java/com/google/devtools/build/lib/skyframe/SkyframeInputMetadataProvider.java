// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.MissingDepExecException;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * An {@link InputMetadataProvider} implementation that requests the metadata of derived artifacts
 * from Skyframe.
 *
 * <p>During input discovery, the action may well legally read scheduling dependencies that are not
 * also inputs. Those are not in the regular input metadata provider (doing so would be a
 * performance issue), so we need to ask those from Skyframe instead. It's not that problematic
 * because they are known to be transitive Skyframe deps, so we can rely on them being present, with
 * one exception (see below) that can be handled without much ceremony.
 *
 * <p>In theory, this would also work for source artifacts. However, the performance ramifications
 * of doing that are unknown.
 */
public class SkyframeInputMetadataProvider implements InputMetadataProvider {
  private final MemoizingEvaluator evaluator;
  private final InputMetadataProvider perBuild;
  private final ConcurrentHashMap<String, ActionInput> seen;

  public SkyframeInputMetadataProvider(
      MemoizingEvaluator evaluator, InputMetadataProvider perBuild) {
    this.evaluator = evaluator;
    this.perBuild = perBuild;
    this.seen = new ConcurrentHashMap<>();
  }

  @Nullable
  @Override
  public FileArtifactValue getInputMetadataChecked(ActionInput input)
      throws InterruptedException, IOException, MissingDepExecException {
    if (!(input instanceof Artifact artifact)) {
      return perBuild.getInputMetadataChecked(input);
    }

    if (artifact.isSourceArtifact()) {
      return perBuild.getInputMetadataChecked(input);
    }

    if (artifact instanceof SpecialArtifact) {
      return null;
    }

    SkyValue value = evaluator.getExistingValue(Artifact.key(artifact));
    if (value == null) {
      // This can only happen if a transitive dependency was rewound but the re-evaluation resulted
      // in an error or the rewinding is in progress. In either case, the InputMetadataProvider is
      // an ActionFileSystem because that's always the case when rewinding is enabled so this code
      // path should never be taken.
      throw new IllegalStateException(
          String.format(
              "Transitive dependency derived artifact '%s' is not in Skyframe",
              artifact.getExecPathString()));
    }

    seen.put(artifact.getExecPathString(), artifact);
    ActionExecutionValue actionExecutionValue = (ActionExecutionValue) value;
    return actionExecutionValue.getExistingFileArtifactValue(artifact);
  }

  @Nullable
  @Override
  public TreeArtifactValue getTreeMetadata(ActionInput input) {
    return null;
  }

  @Nullable
  @Override
  public FilesetOutputTree getFileset(ActionInput input) {
    return null;
  }

  @Override
  public Map<Artifact, FilesetOutputTree> getFilesets() {
    return ImmutableMap.of();
  }

  @Nullable
  @Override
  public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
    return null;
  }

  @Override
  public ImmutableList<RunfilesTree> getRunfilesTrees() {
    return ImmutableList.of();
  }

  @Nullable
  @Override
  public ActionInput getInput(String execPath) {
    ActionInput result = seen.get(execPath);
    if (result == null) {
      result = perBuild.getInput(execPath);
    }

    return result;
  }
}
