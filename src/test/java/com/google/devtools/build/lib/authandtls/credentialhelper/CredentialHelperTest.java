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

package com.google.devtools.build.lib.authandtls.credentialhelper;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.EventBusEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.shell.WindowsSubprocessFactory;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.devtools.build.runfiles.Runfiles;
import java.net.URI;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.SequencedMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CredentialHelperTest {
  static {
    WindowsSubprocessFactory.maybeInstallWindowsSubprocessFactory();
  }

  private static final PathFragment TEST_WORKSPACE_PATH =
      PathFragment.create(System.getenv("TEST_TMPDIR"));
  private static final PathFragment TEST_CREDENTIAL_HELPER_PATH =
      PathFragment.create(
          "io_bazel/src/test/java/com/google/devtools/build/lib/authandtls/credentialhelper/test_credential_helper"
              + (OS.getCurrent() == OS.WINDOWS ? ".exe" : ""));

  private static final Reporter reporter =
      new Reporter(EventBusEventHandler.createWithNewEventBus());

  // Windows VMs in CI can be slow to start the Python-based helper process (wrapped in an exe),
  // so we use a larger timeout (30s) on Windows to avoid flakiness, while keeping it 5s on Linux.
  private static Duration getDefaultTimeout() {
    return OS.getCurrent() == OS.WINDOWS ? Duration.ofSeconds(30) : Duration.ofSeconds(5);
  }

  private GetCredentialsResponse getCredentialsFromHelper(
      String credHelperPath, String uri, ImmutableMap<String, String> env, Duration timeout)
      throws Exception {
    Preconditions.checkNotNull(credHelperPath);
    Preconditions.checkNotNull(uri);
    Preconditions.checkNotNull(env);
    Preconditions.checkNotNull(timeout);

    FileSystem fs = FileSystems.getNativeFileSystem();

    CredentialHelper credentialHelper = new CredentialHelper(fs.getPath(credHelperPath));
    SequencedMap<String, String> clientEnv = new LinkedHashMap<>(System.getenv());
    // Don't cd to the Python credential helper's temporary directory on Windows, which would throw
    // off test assertions. This variable is set to "1" by the surrounding Java test.
    clientEnv.remove("RUN_UNDER_RUNFILES");
    clientEnv.putAll(env);
    return credentialHelper.getCredentials(
        CredentialHelperEnvironment.newBuilder()
            .setEventReporter(reporter)
            .setWorkspacePath(fs.getPath(TEST_WORKSPACE_PATH))
            .setClientEnvironment(() -> ImmutableMap.copyOf(clientEnv))
            .setHelperExecutionTimeout(timeout)
            .build(),
        URI.create(uri));
  }

  private GetCredentialsResponse getCredentialsFromHelper(
      String credHelperPath, String uri, ImmutableMap<String, String> env) throws Exception {
    return getCredentialsFromHelper(credHelperPath, uri, env, getDefaultTimeout());
  }

  private GetCredentialsResponse getCredentialsFromHelper(
      String uri, ImmutableMap<String, String> env, Duration timeout) throws Exception {
    String credHelperPath =
        Runfiles.preload()
            .withSourceRepository("")
            .rlocation(TEST_CREDENTIAL_HELPER_PATH.getPathString());

    return getCredentialsFromHelper(credHelperPath, uri, env, timeout);
  }

  private GetCredentialsResponse getCredentialsFromHelper(
      String uri, ImmutableMap<String, String> env) throws Exception {
    return getCredentialsFromHelper(uri, env, getDefaultTimeout());
  }

  private GetCredentialsResponse getCredentialsFromHelper(String uri) throws Exception {
    Preconditions.checkNotNull(uri);

    return getCredentialsFromHelper(uri, /* env= */ ImmutableMap.of());
  }

  @Test
  public void knownUriWithSingleHeader() throws Exception {
    GetCredentialsResponse response = getCredentialsFromHelper("https://singleheader.example.com");
    assertThat(response.headers()).containsExactly("header1", ImmutableList.of("value1"));
  }

  @Test
  public void knownUriWithMultipleHeaders() throws Exception {
    GetCredentialsResponse response =
        getCredentialsFromHelper("https://multipleheaders.example.com");
    assertThat(response.headers())
        .containsExactly(
            "header1",
            ImmutableList.of("value1"),
            "header2",
            ImmutableList.of("value1", "value2"),
            "header3",
            ImmutableList.of("value1", "value2", "value3"));
  }

  @Test
  public void unknownUri() {
    CredentialHelperException e =
        assertThrows(
            CredentialHelperException.class,
            () -> getCredentialsFromHelper("https://unknown.example.com"));
    assertThat(e).hasMessageThat().contains("Failed to get credentials");
    assertThat(e).hasMessageThat().contains("Unknown uri 'https://unknown.example.com'");
  }

  @Test
  public void credentialHelperOutputsNothing() throws Exception {
    CredentialHelperException e =
        assertThrows(
            CredentialHelperException.class,
            () -> getCredentialsFromHelper("https://printnothing.example.com"));
    assertThat(e).hasMessageThat().contains("Failed to get credentials");
    assertThat(e).hasMessageThat().contains("exited without output");
  }

  @Test
  public void credentialHelperOutputsExtraFields() throws Exception {
    GetCredentialsResponse response = getCredentialsFromHelper("https://extrafields.example.com");
    assertThat(response.headers()).containsExactly("header1", ImmutableList.of("value1"));
  }

  @Test
  public void helperRunsInWorkspace() throws Exception {
    GetCredentialsResponse response = getCredentialsFromHelper("https://cwd.example.com");
    ImmutableMap<String, ImmutableList<String>> headers = response.headers();
    assertThat(PathFragment.create(headers.get("cwd").get(0))).isEqualTo(TEST_WORKSPACE_PATH);
  }

  @Test
  public void helperGetEnvironment() throws Exception {
    GetCredentialsResponse response =
        getCredentialsFromHelper(
            "https://env.example.com", ImmutableMap.of("FOO", "BAR!", "BAR", "123"));
    assertThat(response.headers())
        .containsExactly(
            "foo", ImmutableList.of("BAR!"),
            "bar", ImmutableList.of("123"));
  }

  @Test
  public void helperTimeout() throws Exception {
    CredentialHelperException e =
        assertThrows(
            CredentialHelperException.class,
            () ->
                getCredentialsFromHelper(
                    "https://timeout.example.com",
                    /* env= */ ImmutableMap.of(),
                    /* timeout= */ Duration.ofSeconds(5)));
    assertThat(e).hasMessageThat().contains("Failed to get credentials");
    assertThat(e).hasMessageThat().contains("process timed out");
  }

  @Test
  public void nonExistentHelper() throws Exception {
    CredentialHelperException e =
        assertThrows(
            CredentialHelperException.class,
            () ->
                getCredentialsFromHelper(
                    OS.getCurrent() == OS.WINDOWS ? "C:/no/such/file" : "/no/such/file",
                    "https://timeout.example.com",
                    /* env= */ ImmutableMap.of()));
    assertThat(e).hasMessageThat().contains("Failed to get credentials");
    assertThat(e)
        .hasMessageThat()
        .contains(
            OS.getCurrent().equals(OS.WINDOWS)
                ? "cannot find the file specified"
                : "Cannot run program");
  }

  @Test
  public void argvForRegularHelper() {
    // A helper that is a plain executable is invoked directly on every platform, including one
    // whose name merely contains `.bat`.
    assertThat(CredentialHelper.getArgv(OS.LINUX, "/usr/bin/helper", "get"))
        .containsExactly("/usr/bin/helper", "get")
        .inOrder();
    assertThat(CredentialHelper.getArgv(OS.WINDOWS, "C:/tools/helper.exe", "get"))
        .containsExactly("C:/tools/helper.exe", "get")
        .inOrder();
    assertThat(CredentialHelper.getArgv(OS.WINDOWS, "C:/tools/helper.bat.exe", "get"))
        .containsExactly("C:/tools/helper.bat.exe", "get")
        .inOrder();

    // Batch scripts are only special on Windows.
    assertThat(CredentialHelper.getArgv(OS.LINUX, "/usr/bin/helper.bat", "get"))
        .containsExactly("/usr/bin/helper.bat", "get")
        .inOrder();
  }

  @Test
  public void argvForBatchScriptHelperOnWindows() {
    assertThat(CredentialHelper.getArgv(OS.WINDOWS, "C:/tools/helper.bat", "get"))
        .containsExactly("cmd.exe", "/S", "/D", "/C", "\"C:\\tools\\helper.bat get\"")
        .inOrder();

    // `.cmd` and upper-case extensions are recognized too.
    assertThat(CredentialHelper.getArgv(OS.WINDOWS, "C:/tools/helper.CMD", "get"))
        .containsExactly("cmd.exe", "/S", "/D", "/C", "\"C:\\tools\\helper.CMD get\"")
        .inOrder();

    // A path containing spaces stays in one piece: `/S` strips only the outer quotes, leaving the
    // inner ones for cmd.exe to parse.
    assertThat(CredentialHelper.getArgv(OS.WINDOWS, "C:/Program Files/h.bat", "get"))
        .containsExactly("cmd.exe", "/S", "/D", "/C", "\"\"C:\\Program Files\\h.bat\" get\"")
        .inOrder();
  }

  @Test
  public void batchScriptHelper() throws Exception {
    assumeTrue(OS.getCurrent() == OS.WINDOWS);

    FileSystem fs = FileSystems.getNativeFileSystem();
    // Deliberately place the script in a directory whose name contains a space.
    Path helper = fs.getPath(TEST_WORKSPACE_PATH).getRelative("cred helper/helper.bat");
    helper.getParentDirectory().createDirectoryAndParents();
    // Batch scripts want CRLF line endings.
    FileSystemUtils.writeContentAsLatin1(
        helper,
        String.join(
            "\r\n",
            "@echo off",
            // Verifies that the argument survives the trip through cmd.exe unquoted.
            "if not \"%1\"==\"get\" exit /b 1",
            "echo {\"headers\":{\"header1\":[\"value1\"]}}",
            ""));
    helper.setExecutable(true);

    GetCredentialsResponse response =
        getCredentialsFromHelper(
            helper.getPathString(), "https://example.com", /* env= */ ImmutableMap.of());
    assertThat(response.headers()).containsExactly("header1", ImmutableList.of("value1"));
  }

  @Test
  public void hugePayload() throws Exception {
    // Bazel reads the credential helper stdout/stderr from a pipe, and doesn't start reading
    // until the process terminates. Therefore, a response larger than the pipe buffer causes
    // a deadlock and timeout. This verifies that the pipe is sufficiently large.
    // See https://github.com/bazelbuild/bazel/issues/21287.
    GetCredentialsResponse response = getCredentialsFromHelper("https://hugepayload.example.com");
    assertThat(response.headers()).containsExactly("huge", ImmutableList.of("x".repeat(63 * 1024)));
  }
}
