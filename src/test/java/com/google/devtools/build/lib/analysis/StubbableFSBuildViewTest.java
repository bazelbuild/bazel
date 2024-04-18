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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.util.BuildViewTestBase;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** BuildViewTest where it's possible to stub the FileSystem operations. */
@RunWith(JUnit4.class)
public class StubbableFSBuildViewTest extends BuildViewTestBase {
  @Override
  protected FileSystem createFileSystem() {
    return new StubbableFs(new ManualClock());
  }

  private StubbableFs getStubbableFS() {
    return (StubbableFs) fileSystem;
  }

  // Regression test for b/227641207.
  @Test
  public void testCatastrophicAnalysisErrorAspect_keepGoing_noCrashCatastrophicErrorReported()
      throws Exception {
    // We're expecting failures.
    reporter.removeHandler(failFastHandler);
    Path pathToBuildB = scratch.file("b/BUILD", "cc_library(name='b')");
    scratch.file("a/BUILD", "cc_library(name='a', srcs = ['a.cc'], deps = ['//b:b'])");
    scratch.file("a/a.cc", "");
    scratch.file(
        "a/aspect.bzl",
        """
        def _impl(target, ctx):
            print("This aspect does nothing")
            return struct()

        MyAspect = aspect(implementation = _impl)
        """);
    getStubbableFS().stubFastDigestError(pathToBuildB, new IOException("testException"));
    AnalysisFailureRecorder recorder = new AnalysisFailureRecorder();
    eventBus.register(recorder);

    AnalysisResult result =
        update(
            eventBus,
            defaultFlags().with(Flag.KEEP_GOING),
            ImmutableList.of("a/aspect.bzl%MyAspect"),
            "//a");

    assertThat(result.hasError()).isTrue();
    assertThat(result.getFailureDetail().getMessage())
        .contains("command succeeded, but not all targets were analyzed");
    assertThat(recorder.events).hasSize(1);
    assertThat(
            Iterables.getOnlyElement(recorder.events)
                .getRootCauses()
                .getSingleton()
                .getDetailedExitCode()
                .getFailureDetail()
                .getMessage())
        .contains(
            "Inconsistent filesystem operations. 'stat' said /workspace/b/BUILD is a file but then"
                + " we later encountered error 'testException' which indicates that"
                + " /workspace/b/BUILD is no longer a file.");
  }

  // Regression test for b/227641207.
  @Test
  public void testCatastrophicAnalysisError_keepGoing_noCrashCatastrophicErrorReported()
      throws Exception {
    // We're expecting failures.
    reporter.removeHandler(failFastHandler);
    Path pathToBuildB = scratch.file("b/BUILD", "cc_library(name='b')");
    scratch.file("a/BUILD", "cc_library(name='a', srcs = ['a.cc'], deps = ['//b:b'])");
    scratch.file("a/a.cc", "");
    getStubbableFS().stubFastDigestError(pathToBuildB, new IOException("testExeception"));
    AnalysisFailureRecorder recorder = new AnalysisFailureRecorder();
    eventBus.register(recorder);

    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//a");

    assertThat(result.hasError()).isTrue();
    assertThat(result.getFailureDetail().getMessage())
        .contains("command succeeded, but not all targets were analyzed");
    assertThat(recorder.events).hasSize(1);
    assertThat(
            Iterables.getOnlyElement(recorder.events)
                .getRootCauses()
                .getSingleton()
                .getDetailedExitCode()
                .getFailureDetail()
                .getMessage())
        .contains(
            "Inconsistent filesystem operations. 'stat' said /workspace/b/BUILD is a file but then"
                + " we later encountered error 'testExeception' which indicates that"
                + " /workspace/b/BUILD is no longer a file.");
  }

  private static class StubbableFs extends InMemoryFileSystem {

    private final Map<PathFragment, IOException> stubbedFastDigestErrors = Maps.newHashMap();

    StubbableFs(ManualClock manualClock) {
      super(manualClock, DigestHashFunction.SHA256);
    }

    void stubFastDigestError(Path path, IOException error) {
      stubbedFastDigestErrors.put(path.asFragment(), error);
    }

    @Override
    @SuppressWarnings("UnsynchronizedOverridesSynchronized")
    protected byte[] getFastDigest(PathFragment path) throws IOException {
      if (stubbedFastDigestErrors.containsKey(path)) {
        throw stubbedFastDigestErrors.get(path);
      }
      return getDigest(path);
    }
  }
}
