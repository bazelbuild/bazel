// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that concern the transitive closure of data dependencies. Regression testing for bug
 * 1022571.
 */
@RunWith(JUnit4.class)
public abstract class TransitiveDataDependencyTest extends BuildIntegrationTestCase {

  /**
   * Hook for subclasses to define which executor we use.  (The two concrete
   * subclasses, {Sequential,Parallel}TransitiveDataDependencyTest are at the bottom of this
   * source file.)
   */
  protected abstract int numJobs();

  @Before
  public final void addJobNumberOption() throws Exception  {
    addOptions("--jobs", "" + numJobs());
  }

  private void assertSameConfiguredTarget(String label) throws Exception {
    assertThat(getOnlyElement(getResult().getSuccessfulTargets()))
        .isSameInstanceAs(getConfiguredTarget(label));
  }

  @Test
  public void testTransitiveDataDepIsBuilt() throws Exception {
    write(
        "data/BUILD",
        """
        cc_library(
            name = "needsdata",
            data = [":data_bin"],
        )

        cc_binary(
            name = "data_bin",
            srcs = ["data_bin.c"],
        )
        """);
    write("data/data_bin.c", "int main() { return 0; }");

    buildTarget("//data:needsdata");
    ConfiguredTarget dataLibTarget = getConfiguredTarget("//data:data_bin");
    assertThat(getFilesToBuild(dataLibTarget).toList()).isNotEmpty();
    for (Artifact dataOut : getFilesToBuild(dataLibTarget).toList()) {
      assertWithMessage("Missing output: " + dataOut.getPath())
          .that(dataOut.getPath().exists())
          .isTrue();
    }
    assertSameConfiguredTarget("//data:needsdata");
  }

  @Test
  public void testMissingInputFile() throws Exception {
    write("data/BUILD",
        "cc_library(name = 'needsdata', data = [':data_file'])");

    RecordingOutErr recOutErr = new RecordingOutErr();
    OutErr origOutErr = this.outErr;
    this.outErr = recOutErr;

    // Remove this flag after fixing:
    // "Remove source artifacts from top-level artifacts in SkyframeExecutor#buildArtifacts"
    // We are adding information about //data:needdata to error message only if
    // ActionExecutionFunction has requested that missing artifact. But there is small chance
    // that ArtifactFunction of that missing artifact has thrown exception before that request.
    // In case of keep_going we can be sure that ActionExecutionFunction has made request.
    addOptions("--keep_going");
    try {
      buildTarget("//data:needsdata");
      fail();
    } catch (BuildFailedException e) {
      assertThat(recOutErr.errAsLatin1())
          .containsMatch("//data:needsdata: missing input file '//data:data_file'");
    } finally {
      this.outErr = origOutErr;
    }
    assertThat(getResult().getSuccess()).isFalse();
    assertThat(getResult().getSuccessfulTargets()).isEmpty();
  }

  @Test
  public void testMissingExportsFiles() throws Exception {
    write("data/BUILD", "exports_files(['nosuchfile'])");

    RecordingOutErr recOutErr = new RecordingOutErr();
    OutErr origOutErr = this.outErr;
    this.outErr = recOutErr;

    try {
      buildTarget("//data:nosuchfile");
      fail();
    } catch (BuildFailedException e) {
      assertThat(recOutErr.errAsLatin1()).containsMatch("missing input file '//data:nosuchfile'");
    } finally {
      this.outErr = origOutErr;
    }
    assertThat(getResult().getSuccess()).isFalse();
    assertThat(getResult().getSuccessfulTargets()).isEmpty();
  }


  @Test
  public void testMissingInputFilesKeepGoing() throws Exception {
    write(
        "data/BUILD",
        """
        # Comment line
        cc_library(
            name = "needsdata1",
            data = [":data_file1"],
        )

        cc_library(
            name = "needsdata2",
            data = [":data_file2"],
        )
        """);
    write("data/data_file2", "data_file2 exists");

    RecordingOutErr recOutErr = new RecordingOutErr();
    OutErr origOutErr = this.outErr;
    this.outErr = recOutErr;

    addOptions("--keep_going");
    try {
      buildTarget("//data:needsdata1", "//data:needsdata2");
      fail();
    } catch (BuildFailedException expected) {
      assertThat(recOutErr.errAsLatin1())
          .containsMatch(
              "data/BUILD:2:1: //data:needsdata1: missing input file '//data:data_file1'");
    } finally {
      this.outErr = origOutErr;
    }

    assertThat(getResult().getSuccess()).isFalse();
    assertSameConfiguredTarget("//data:needsdata2");
  }

  // Concrete implementations of this abstract test:

  /** Tests with 1 job. */
  @RunWith(JUnit4.class)
  public static class SequentialTransitiveDataDependencyTest extends TransitiveDataDependencyTest {
    @Override
    protected int numJobs() { return 0; }
  }

  /** Tests with 100 jobs. */
  @RunWith(JUnit4.class)
  public static class ParallelTransitiveDataDependencyTest extends TransitiveDataDependencyTest {
    @Override
    protected int numJobs() { return 100; }
  }
}
