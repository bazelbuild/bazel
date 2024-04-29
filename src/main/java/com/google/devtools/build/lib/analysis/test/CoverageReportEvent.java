// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.CoverageReport;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.File;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import java.util.Collection;

/** Inform BEP consumers that the coverage report is available. */
public class CoverageReportEvent implements BuildEvent {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final ImmutableSet<Artifact> coverageReport;

  public CoverageReportEvent(ImmutableSet<Artifact> coverageReport) {
    this.coverageReport = coverageReport;
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context)
      throws InterruptedException {
    ImmutableList.Builder<File> reportFiles = ImmutableList.builder();
    for (Artifact artifact : coverageReport) {
      String uri = context.pathConverter().apply(artifact.getPath());
      if (uri != null) {
        reportFiles.add(File.newBuilder().setUri(uri).build());
      } else {
        logger.atInfo().log("Dropped unfinished upload: %s", artifact.getPath());
      }
    }
    return GenericBuildEvent.protoChaining(this)
        .setCoverageReport(CoverageReport.newBuilder().addAllCoverageReport(reportFiles.build()))
        .build();
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.coverageReport();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public Collection<LocalFile> referencedLocalFiles() {
    return coverageReport.stream()
        .map(
            artifact ->
                new LocalFile(
                    artifact.getPath(),
                    LocalFile.LocalFileType.COVERAGE_OUTPUT,
                    artifact,
                    /* artifactMetadata= */ null))
        .collect(toImmutableList());
  }
}
