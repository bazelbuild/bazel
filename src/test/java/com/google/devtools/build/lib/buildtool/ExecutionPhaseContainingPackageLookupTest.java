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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test case for crashes caused by exceptions happening during package lookups during the
 * execution phase under certain conditions. To trigger a failure we must:
 *
 * <ul>
 * <li>have a populated action cache on disk but no in memory state (ie: build after restart)</li>
 * <li>have a build that does include scanning where the includes are not explicitly declared
 *      and reside in a directory below a package (common)</li>
 * <li>have some sort of filesystem event that would trigger exceptions between the loading
 *      and execution phase (ie: network disconnecting, credentials expiring, etc)</li>
 * </ul>
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class ExecutionPhaseContainingPackageLookupTest extends IoHookTestCase {

  @Test
  public void testIOExceptionDuringExecutionPhaseContainingPackageLookup() throws Exception {
    // Our main code that includes the include we'll be scanning for during the exec phase.
    write(
        "foo/BUILD",
        "cc_library(name = 'foo', srcs = ['foo.cc'], deps = ['//bar', '//notfinished'])");
    write(
        "foo/foo.cc",
        "#include \"bar/includes/include.h\"",
        "#include \"notfinished/includes/include.h\"",
        "int main() { return -1; }");

    // Dummy package containing a not immediately obvious include.
    write("bar/BUILD",
        "cc_library(name = 'bar', srcs = [], hdrs_check = 'loose',",
        "           visibility = ['//visibility:public'])");
    write("bar/includes/include.h");

    // Package that won't have its BUILD file looked up when the second build aborts.
    write(
        "notfinished/BUILD",
        "cc_library(name = 'notfinished', srcs = [], hdrs_check = 'loose',",
        "           visibility = ['//visibility:public'])");
    write("notfinished/includes/include.h");

    // Build and toss the analysis cache, imperative to make sure we go to disk in
    // the next build.
    addOptions("--discard_analysis_cache", "--features=cc_include_scanning");
    buildTarget("//foo:foo");

    final AtomicBoolean interrupted = new AtomicBoolean(true);

    // GoogleBuildIntegrationTestCase installs a remote logger that calls Runtime.halt, and
    // SkyframeBuilder calls logToRemote when there's a BuildFileNotFoundException, and the code
    // below triggers a BuildFileNotFoundException. Previously, the ActionExecutionFunction was
    // rethrowing that exception as a ActionExecutionFunctionException during error bubbling, but
    // it now ignores it, which triggers the code path in SkyframeBuilder. However, the test runner
    // catches Runtime.halt, and throws an exception instead. We just overwrite the remote logger
    // with one that silently ignores the log.
    LoggingUtil.installRemoteLoggerForTesting(Futures.immediateFuture(Logger.getAnonymousLogger()));

    // Now we want any lookup for "bar/includes/BUILD" to throw an IOException.
    setListener(
        new FileListener() {
          @Override
          public void handle(PathOp op, Path path) throws IOException {
            if (path.getPathString().contains("notfinished/includes/BUILD")) {
              // Make sure we don't have the PackageLookup results for notfinished/includes before
              // we throw an IOException below. This tests that we'll still throw the proper
              // exception even without all the deps being present.
              try {
                Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                interrupted.set(false);
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              }
            } else if (path.getPathString().contains("bar/includes/BUILD")) {
              throw new IOException("FAIIIIIILLLLLLLLL");
            }
          }
        });

    BuildFailedException bfe =
        assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
    assertThat(bfe).hasMessageThat().contains("no such package 'bar/includes'");
    assertThat(interrupted.get()).isTrue();
  }

}
