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

package com.google.devtools.build.lib.rules.test;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;

import java.util.Collection;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A factory class to create coverage report actions.
 */
public interface CoverageReportActionFactory {

  /**
   * Returns a coverage report Action. May return null if it's not necessary to create
   * such an Action based on the input parameters and some other data available to
   * the factory implementation, such as command line arguments.
   */
  @Nullable
  public Action createCoverageReportAction(Collection<ConfiguredTarget> targetsToTest,
      Set<Artifact> baselineCoverageArtifacts,
      ArtifactFactory artifactFactory, ArtifactOwner artifactOwner);
}