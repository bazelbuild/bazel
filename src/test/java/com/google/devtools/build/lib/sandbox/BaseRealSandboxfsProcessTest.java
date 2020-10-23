// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.After;
import org.junit.Before;

/**
 * Common code to unit test {@link RealSandboxfsProcess}.
 *
 * <p>These tests validate the communication protocol between Bazel and a sandboxfs but do so using
 * golden data. They are meant to smoke-test changes to the Bazel codebase against all supported
 * sandboxfs versions but cannot guarantee that the integration with a real sandboxfs binary work.
 */
public abstract class BaseRealSandboxfsProcessTest {

  /** Path to the mount point passed to sandboxfs. Not accessed, so it's not even created. */
  static final String FAKE_MOUNT_POINT = "/non-existent/mount/point";

  /** Sandboxfs version to return to Bazel when queried. */
  private final String version;

  /**
   * Expected mount arguments for each {@link #createAndStartFakeSandboxfs} call. Supplied as a
   * single string with all arguments concatenated as seen by sandboxfs.
   */
  private final String expectedArgs;

  private FileSystem fileSystem;
  private Path tmpDir;

  // Initialized via createAndStartFakeSandboxfs and checked by verifyFakeSandboxfsExecution.
  private Path capturedArgs;
  private Path capturedRequests;

  BaseRealSandboxfsProcessTest(String version, String expectedArgs) {
    this.version = version;
    this.expectedArgs = expectedArgs;
  }

  @Before
  public void setUp() throws Exception {
    fileSystem = new JavaIoFileSystem(DigestHashFunction.SHA256);
    tmpDir = fileSystem.getPath(System.getenv("TEST_TMPDIR")).getRelative("test");
    tmpDir.createDirectory();
  }

  @After
  public void tearDown() throws Exception {
    tmpDir.deleteTree();
  }

  /**
   * Starts a sandboxfs instance using a fake binary that captures all received requests and yields
   * mock responses.
   *
   * @param responses the mock responses to return to Bazel when issuing requests, broken down by
   *     line printed to stdout
   * @return a sandboxfs process handler
   * @throws IOException if the fake sandboxfs cannot be prepared or started
   */
  SandboxfsProcess createAndStartFakeSandboxfs(List<String> responses) throws IOException {
    capturedArgs = tmpDir.getRelative("captured-args");
    capturedRequests = tmpDir.getRelative("captured-requests");

    Path fakeSandboxfs = tmpDir.getRelative("fake-sandboxfs");
    try (PrintWriter writer =
        new PrintWriter(
            new BufferedWriter(
                new OutputStreamWriter(fakeSandboxfs.getOutputStream(), StandardCharsets.UTF_8)))) {
      writer.println("#! /bin/bash");

      // Ignore requests for termination. The real sandboxfs process must be sent a SIGTERM to stop
      // serving, but in our case we want to terminate cleanly after waiting for all input to be
      // recorded.
      writer.println("trap '' TERM;");

      // Handle a --version invocation and exit quickly, which is a prerequisite for the mount call.
      writer.println("if [ \"${*}\" = \"--version\" ]; then");
      writer.println("  echo sandboxfs " + version + ";");
      writer.println("  exit 0;");
      writer.println("fi;");

      // Capture all arguments for later inspection.
      writer.println("for arg in \"${@}\"; do");
      writer.println("  echo \"${arg}\" >>" + capturedArgs + ";");
      writer.println("done;");

      // Attempt to "parse" requests coming through stdin by just counting brace pairs, assuming
      // that the input is composed of a stream of JSON objects. Then, for each request, emit one
      // response.
      //
      // We must do this because the unordered response processor required to parse 0.2.0 output
      // expects responses to come only after their requests have been issued. Ideally we'd match
      // our mock responses to specific requests to allow for testing of unordered responses, but
      // for now assume all requests and responses in the test are correctly ordered.
      //
      // TODO(jmmv): This has become pretty awful. Should rethink unit testing.
      for (String response : responses) {
        writer.println("braces=0; started=no");
        writer.println("while read -d '' -n 1 ch; do");
        writer.println("  case \"${ch}\" in");
        writer.println("    '{') braces=$((braces + 1)); started=yes ;;");
        writer.println("    '[') braces=$((braces + 1)); started=yes ;;");
        writer.println("    ']') braces=$((braces - 1)) ;;");
        writer.println("    '}') braces=$((braces - 1)) ;;");
        writer.println("  esac");
        writer.println("  [[ \"${ch}\" != '' ]] || ch='\n'");
        writer.println("  printf '%c' \"${ch}\" >>" + capturedRequests);
        writer.println("  if [[ \"${started}\" = yes && \"${braces}\" -eq 0 ]]; then");
        writer.println("    echo '" + response + "';");
        writer.println("    break;");
        writer.println("  fi");
        writer.println("done");
      }

      // Capture any stray requests not expected by the test data.
      writer.println("cat >>" + capturedRequests);
    }
    fakeSandboxfs.setExecutable(true);

    return RealSandboxfsProcess.mount(
        fakeSandboxfs.asFragment(),
        fileSystem.getPath(FAKE_MOUNT_POINT),
        tmpDir.getRelative("log"));
  }

  /**
   * Checks that the given sandboxfs process behaved as expected.
   *
   * @param process the sandboxfs instance to stop and verify, which must have been previously
   *     started by {@link #createAndStartFakeSandboxfs}
   * @param expectedRequests a flat string containing all requests given to sandboxfs (i.e. the raw
   *     contents of its stdin)
   * @throws IOException if the fake sandboxfs instance cannot be stopped or if there is a problem
   *     reading the captured data
   */
  void verifyFakeSandboxfsExecution(SandboxfsProcess process, String expectedRequests)
      throws IOException {
    process.destroy();

    String args = FileSystemUtils.readContent(capturedArgs, StandardCharsets.UTF_8);
    assertThat(args).isEqualTo(expectedArgs);

    String requests = FileSystemUtils.readContent(capturedRequests, StandardCharsets.UTF_8);
    assertThat(requests).isEqualTo(expectedRequests);
  }
}
