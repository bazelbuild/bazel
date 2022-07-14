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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.time.Duration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CredentialHelperTest {
  private static final PathFragment TEST_WORKSPACE_PATH =
      PathFragment.create(System.getenv("TEST_TMPDIR"));
  private static final PathFragment TEST_CREDENTIAL_HELPER_DIR =
      PathFragment.create(
          "io_bazel/src/test/java/com/google/devtools/build/lib/authandtls/credentialhelper");

  private static final Runfiles runfiles;

  static {
    try {
      runfiles = Runfiles.create();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static final Reporter reporter = new Reporter(new EventBus());

  private GetCredentialsResponse getCredentialsFromHelper(
      String uri, ImmutableMap<String, String> env) throws Exception {
    Preconditions.checkNotNull(uri);
    Preconditions.checkNotNull(env);

    FileSystem filesystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    PathFragment credentialHelperPath =
        TEST_CREDENTIAL_HELPER_DIR.getChild(
            OS.getCurrent() == OS.WINDOWS
                ? "test_credential_helper.exe"
                : "test_credential_helper");
    CredentialHelper credentialHelper =
        new CredentialHelper(
            filesystem.getPath(runfiles.rlocation(credentialHelperPath.getSafePathString())));
    return credentialHelper.getCredentials(
        CredentialHelperEnvironment.newBuilder()
            .setEventReporter(reporter)
            .setWorkspacePath(filesystem.getPath(TEST_WORKSPACE_PATH))
            .setClientEnvironment(env)
            .setHelperExecutionTimeout(Duration.ofSeconds(5))
            .build(),
        URI.create(uri));
  }

  private GetCredentialsResponse getCredentialsFromHelper(String uri) throws Exception {
    Preconditions.checkNotNull(uri);

    return getCredentialsFromHelper(uri, ImmutableMap.of());
  }

  @Test
  public void knownUriWithSingleHeader() throws Exception {
    GetCredentialsResponse response = getCredentialsFromHelper("https://singleheader.example.com");
    assertThat(response.getHeaders())
        .containsExactlyEntriesIn(
            ImmutableMap.<String, ImmutableList<String>>builder()
                .put("header1", ImmutableList.of("value1"))
                .build());
  }

  @Test
  public void knownUriWithMultipleHeaders() throws Exception {
    GetCredentialsResponse response =
        getCredentialsFromHelper("https://multipleheaders.example.com");
    assertThat(response.getHeaders())
        .containsExactlyEntriesIn(
            ImmutableMap.<String, ImmutableList<String>>builder()
                .put("header1", ImmutableList.of("value1"))
                .put("header2", ImmutableList.of("value1", "value2"))
                .put("header3", ImmutableList.of("value1", "value2", "value3"))
                .build());
  }

  @Test
  public void unknownUri() {
    IOException ioException =
        assertThrows(
            IOException.class, () -> getCredentialsFromHelper("https://unknown.example.com"));
    assertThat(ioException).hasMessageThat().contains("Unknown uri 'https://unknown.example.com'");
  }

  @Test
  public void credentialHelperOutputsNothing() throws Exception {
    IOException ioException =
        assertThrows(
            IOException.class, () -> getCredentialsFromHelper("https://printnothing.example.com"));
    assertThat(ioException).hasMessageThat().contains("exited without output");
  }

  @Test
  public void credentialHelperOutputsExtraFields() throws Exception {
    GetCredentialsResponse response = getCredentialsFromHelper("https://extrafields.example.com");
    assertThat(response.getHeaders())
        .containsExactlyEntriesIn(
            ImmutableMap.<String, ImmutableList<String>>builder()
                .put("header1", ImmutableList.of("value1"))
                .build());
  }

  @Test
  public void helperRunsInWorkspace() throws Exception {
    GetCredentialsResponse response = getCredentialsFromHelper("https://cwd.example.com");
    ImmutableMap<String, ImmutableList<String>> headers = response.getHeaders();
    assertThat(PathFragment.create(headers.get("cwd").get(0))).isEqualTo(TEST_WORKSPACE_PATH);
  }

  @Test
  public void helperGetEnvironment() throws Exception {
    GetCredentialsResponse response =
        getCredentialsFromHelper(
            "https://env.example.com", ImmutableMap.of("FOO", "BAR!", "BAR", "123"));
    assertThat(response.getHeaders())
        .containsExactlyEntriesIn(
            ImmutableMap.<String, ImmutableList<String>>builder()
                .put("foo", ImmutableList.of("BAR!"))
                .put("bar", ImmutableList.of("123"))
                .build());
  }

  @Test
  public void helperTimeout() throws Exception {
    IOException ioException =
        assertThrows(
            IOException.class, () -> getCredentialsFromHelper("https://timeout.example.com"));
    assertThat(ioException).hasMessageThat().contains("process did not exit");
  }
}
