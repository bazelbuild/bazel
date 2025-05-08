// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.cquery;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Cquery output formatter that prints the set of output files advertised by the matched targets.
 */
public class FilesOutputFormatterCallback extends CqueryThreadsafeCallback {

  private final TopLevelArtifactContext topLevelArtifactContext;

  FilesOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<CqueryNode> accessor,
      TopLevelArtifactContext topLevelArtifactContext) {
    // Different targets may provide the same artifact, so we deduplicate the collection of all
    // results at the end.
    super(eventHandler, options, out, skyframeExecutor, accessor, /*uniquifyResults=*/ true);
    this.topLevelArtifactContext = topLevelArtifactContext;
  }

  @Override
  public String getName() {
    return "files";
  }

  @Override
  public void processOutput(Iterable<CqueryNode> partialResult)
      throws IOException, InterruptedException {
    for (CqueryNode target : partialResult) {
      if (!(target instanceof ConfiguredTarget cf)
          || (!TopLevelArtifactHelper.shouldConsiderForDisplay(target)
              && !(target instanceof InputFileConfiguredTarget))) {
        continue;
      }

      TopLevelArtifactHelper.getAllArtifactsToBuild(cf, topLevelArtifactContext)
          .getImportantArtifacts()
          .toList()
          .stream()
          .filter(
              artifact ->
                  TopLevelArtifactHelper.shouldDisplay(artifact) || artifact.isSourceArtifact())
          .map(Artifact::getExecPathString)
          .forEach(this::addResult);
    }
  }
}
