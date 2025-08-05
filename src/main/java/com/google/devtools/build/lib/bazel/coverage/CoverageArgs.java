// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.coverage;

import static java.util.Objects.requireNonNull;

import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import javax.annotation.Nullable;

/**
 * A value class that holds arguments for {@link CoverageReportActionBuilder#CoverageHelper}
 * methods.
 */
public record CoverageArgs(
    BlazeDirectories directories,
    NestedSet<Artifact> coverageArtifacts,
    Artifact lcovArtifact,
    ArtifactFactory factory,
    ArtifactOwner artifactOwner,
    FilesToRunProvider reportGenerator,
    String workspaceName,
    @Nullable Artifact htmlReport,
    ActionOwner actionOwner) {
  public CoverageArgs {
    requireNonNull(directories, "directories");
    requireNonNull(coverageArtifacts, "coverageArtifacts");
    requireNonNull(lcovArtifact, "lcovArtifact");
    requireNonNull(factory, "factory");
    requireNonNull(artifactOwner, "artifactOwner");
    requireNonNull(reportGenerator, "reportGenerator");
    requireNonNull(workspaceName, "workspaceName");
    requireNonNull(actionOwner);
  }
}
