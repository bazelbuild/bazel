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
import static com.google.devtools.build.lib.analysis.TargetCompleteEvent.newFileFromArtifact;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.EventReportingArtifacts.ReportedArtifacts;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.File;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TargetCompleteEvent}. */
@RunWith(JUnit4.class)
public class TargetCompleteEventTest extends AnalysisTestCase {

  /** Regression test for b/165671166. */
  @Test
  public void testFileProtoFromArtifactReencodesAsUtf8() throws Exception {
    if (OS.getCurrent() == OS.WINDOWS) {
      // Windows filesystems return paths with wide characters and don't suffer from the current
      // workaround where arbitrary bytes are represented to Java as Latin-1.
      return;
    }
    scratch.file("sh/BUILD", "filegroup(name = 'globby', srcs = glob(['dir/*']))");
    // Bytes are UTF-8 encoding of: sh/dir/圖片
    byte[] filenameBytes = {
      0x73, 0x68, 0x2f, 0x64, 0x69, 0x72, 0x2f, -27, -100, -106, -25, -119, -121
    };
    String utf8InLatin1FileName = new String(filenameBytes, ISO_8859_1);
    scratch.file(utf8InLatin1FileName, "content does not matter");
    AnalysisResult result = update("//sh:globby");
    ConfiguredTarget ct = Iterables.getOnlyElement(result.getTargetsToBuild());
    TargetAndConfiguration tac = Iterables.getOnlyElement(result.getTopLevelTargetsWithConfigs());
    ConfiguredTargetAndData ctAndData =
        new ConfiguredTargetAndData(ct, tac.getTarget(), tac.getConfiguration(), null);
    TopLevelArtifactContext context =
        new TopLevelArtifactContext(false, false, false, OutputGroupInfo.DEFAULT_GROUPS);
    ArtifactsToBuild artifactsToBuild = TopLevelArtifactHelper.getAllArtifactsToBuild(ct, context);
    TargetCompleteEvent event =
        TargetCompleteEvent.successfulBuild(
            ctAndData,
            CompletionContext.FAILED_COMPLETION_CTX,
            artifactsToBuild.getAllArtifactsByOutputGroup(),
            /*announceTargetSummary=*/ false);

    ArrayList<File> fileProtos = new ArrayList<>();
    ReportedArtifacts reportedArtifacts = event.reportedArtifacts();
    for (NestedSet<Artifact> artifactSet : reportedArtifacts.artifacts) {
      artifactSet.forEachElement(
          o -> true,
          a -> fileProtos.add(newFileFromArtifact(null, a, PathFragment.EMPTY_FRAGMENT).build()));
    }
    // Bytes are the same but the encoding is actually UTF-8 as required of a protobuf string.
    String utf8FileName = new String(filenameBytes, UTF_8);
    assertThat(fileProtos).hasSize(1);
    assertThat(fileProtos.get(0).getName()).isEqualTo(utf8FileName);
  }
}
