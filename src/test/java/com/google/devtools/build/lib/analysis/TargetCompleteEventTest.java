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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TargetCompleteEvent}. */
@RunWith(JUnit4.class)
public class TargetCompleteEventTest extends AnalysisTestCase {
  /** Regression test for b/111653523. */
  @Test
  public void testReferencedLocalFilesIncludesBaselineCoverage() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'Example', srcs = ['Example.java'])");
    useConfiguration("--collect_code_coverage");
    AnalysisResult result = update("//java/a:Example");
    ConfiguredTarget ct = Iterables.getOnlyElement(result.getTargetsToBuild());
    TargetAndConfiguration tac = Iterables.getOnlyElement(result.getTopLevelTargetsWithConfigs());
    ConfiguredTargetAndData ctAndData =
        new ConfiguredTargetAndData(ct, tac.getTarget(), tac.getConfiguration(), null);
    TopLevelArtifactContext context =
        new TopLevelArtifactContext(false, false, OutputGroupInfo.DEFAULT_GROUPS);
    ArtifactsToBuild artifactsToBuild =
        TopLevelArtifactHelper.getAllArtifactsToBuild(ct, context);
    TargetCompleteEvent event =
        TargetCompleteEvent.successfulBuild(
            ctAndData,
            CompletionContext.FAILED_COMPLETION_CTX,
            artifactsToBuild.getAllArtifactsByOutputGroup());
    assertThat(event.referencedLocalFiles())
        .contains(
            new BuildEvent.LocalFile(
                tac.getConfiguration()
                    .getTestLogsDirectory(RepositoryName.DEFAULT)
                    .getRoot()
                    .asPath()
                    .getRelative("java/a/Example/baseline_coverage.dat"),
                LocalFileType.COVERAGE_OUTPUT));
  }
}
