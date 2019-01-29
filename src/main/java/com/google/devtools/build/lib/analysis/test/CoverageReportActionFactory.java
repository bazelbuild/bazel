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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.events.EventHandler;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * A factory class to create coverage report actions.
 */
public interface CoverageReportActionFactory {
  /**
   * Wraps the necessary actions to get a coverage report as well as the final output artifacts.
   * The lcovWriteAction creates a file containing a set of lcov files. This file is used as an
   * input artifact for coverageReportAction. We are only interested about the output artifacts from
   * coverageReportAction.
   */
  public static final class CoverageReportActionsWrapper {
    private final ActionAnalysisMetadata lcovWriteAction;
    private final ActionAnalysisMetadata coverageReportAction;

    public CoverageReportActionsWrapper (
        ActionAnalysisMetadata lcovWriteAction, ActionAnalysisMetadata coverageReportAction) {
      this.lcovWriteAction = lcovWriteAction;
      this.coverageReportAction = coverageReportAction;
    }

    public ActionAnalysisMetadata getCoverageReportAction() {
      return coverageReportAction;
    }

    public ImmutableList<ActionAnalysisMetadata> getActions() {
      return ImmutableList.of(lcovWriteAction, coverageReportAction);
    }

    public ImmutableSet<Artifact> getCoverageOutputs() {
      return coverageReportAction.getOutputs();
    }
  }

  /**
   * Returns a CoverageReportActionsWrapper. May return null if it's not necessary to create such
   * Actions based on the input parameters and some other data available to the factory
   * implementation, such as command line options.
   */
  @Nullable
  CoverageReportActionsWrapper createCoverageReportActionsWrapper(
      EventHandler eventHandler,
      EventBus eventBus,
      BlazeDirectories directories,
      Collection<ConfiguredTarget> targetsToTest,
      Iterable<Artifact> baselineCoverageArtifacts,
      ArtifactFactory artifactFactory,
      ArtifactOwner artifactOwner,
      String workspaceName);
}
