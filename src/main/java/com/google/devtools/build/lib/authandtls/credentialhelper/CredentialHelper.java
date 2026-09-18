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

import static com.google.devtools.build.lib.profiler.ProfilerTask.CREDENTIAL_HELPER;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.CharMatcher;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.Immutable;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.net.URI;
import java.util.Locale;
import java.util.Objects;

/** Wraps an external tool used to obtain credentials. */
@Immutable
public final class CredentialHelper {
  private static final Gson GSON = new Gson();

  // `Path` is immutable, but not annotated.
  @SuppressWarnings("Immutable")
  private final Path path;

  CredentialHelper(Path path) {
    this.path = Preconditions.checkNotNull(path);
  }

  @VisibleForTesting
  public Path getPath() {
    return path;
  }

  /**
   * Fetches credentials for the specified {@link URI} by invoking the credential helper as
   * subprocess according to the <a
   * href="https://github.com/bazelbuild/proposals/blob/main/designs/2022-06-07-bazel-credential-helpers.md">credential
   * helper protocol</a>.
   *
   * @param environment The environment to run the subprocess in.
   * @param uri The {@link URI} to fetch credentials for.
   * @return The response from the subprocess.
   */
  public GetCredentialsResponse getCredentials(CredentialHelperEnvironment environment, URI uri)
      throws IOException {
    Preconditions.checkNotNull(environment);
    Preconditions.checkNotNull(uri);

    try (SilentCloseable c =
        Profiler.instance().profile(CREDENTIAL_HELPER, "calling credential helper")) {
      Subprocess process;

      try {
        process = spawnSubprocess(environment, "get");
      } catch (IOException e) {
        throw new CredentialHelperException(
            String.format(
                Locale.US,
                "Failed to get credentials for '%s' from helper '%s': %s",
                uri,
                path,
                e.getMessage()));
      }

      try (Reader stdout = new InputStreamReader(process.getInputStream(), UTF_8);
          Reader stderr = new InputStreamReader(process.getErrorStream(), UTF_8)) {
        try (Writer stdin = new OutputStreamWriter(process.getOutputStream(), UTF_8)) {
          GSON.toJson(GetCredentialsRequest.newBuilder().setUri(uri).build(), stdin);
        } catch (IOException e) {
          // This can happen if the helper prints a static set of credentials without reading from
          // stdin (e.g., with a simple shell script running `echo "{...}"`). This is fine to
          // ignore.
        }

        try {
          process.waitFor();
        } catch (InterruptedException e) {
          throw new CredentialHelperException(
              String.format(
                  Locale.US,
                  "Failed to get credentials for '%s' from helper '%s': process was interrupted",
                  uri,
                  path));
        }

        if (process.timedout()) {
          throw new CredentialHelperException(
              String.format(
                  Locale.US,
                  "Failed to get credentials for '%s' from helper '%s': process timed out",
                  uri,
                  path));
        }
        if (process.exitValue() != 0) {
          throw new CredentialHelperException(
              String.format(
                  Locale.US,
                  "Failed to get credentials for '%s' from helper '%s': process exited with code"
                      + " %d. stderr: %s",
                  uri,
                  path,
                  process.exitValue(),
                  CharStreams.toString(stderr)));
        }

        try {
          GetCredentialsResponse response = GSON.fromJson(stdout, GetCredentialsResponse.class);
          if (response == null) {
            throw new CredentialHelperException(
                String.format(
                    Locale.US,
                    "Failed to get credentials for '%s' from helper '%s': process exited without"
                        + " output. stderr: %s",
                    uri,
                    path,
                    CharStreams.toString(stderr)));
          }
          return response;
        } catch (JsonSyntaxException e) {
          throw new CredentialHelperException(
              String.format(
                  Locale.US,
                  "Failed to get credentials for '%s' from helper '%s': error parsing output."
                      + " stderr: %s",
                  uri,
                  path,
                  CharStreams.toString(stderr)),
              e);
        }
      }
    }
  }

  private Subprocess spawnSubprocess(CredentialHelperEnvironment environment, String... args)
      throws IOException {
    Preconditions.checkNotNull(environment);
    Preconditions.checkNotNull(args);

    ImmutableMap<String, String> clientEnv = environment.clientEnvironment().get();
    return new SubprocessBuilder(clientEnv)
        .setArgv(getArgv(OS.getCurrent(), path.getPathString(), args))
        .setWorkingDirectory(
            environment.workspacePath() != null ? environment.workspacePath().getPathFile() : null)
        .setEnv(clientEnv)
        .setTimeoutMillis(environment.helperExecutionTimeout().toMillis())
        .start();
  }

  /**
   * Returns the command line used to invoke the helper.
   *
   * <p>Batch scripts get special treatment on Windows: {@code CreateProcess}, which Bazel uses to
   * spawn subprocesses there, can only start executable images, so a {@code .bat} or {@code .cmd}
   * file has to be handed to the command interpreter instead. This lets a credential helper ship as
   * a small wrapper script on Windows, the way a shell script can be used everywhere else.
   */
  @VisibleForTesting
  static ImmutableList<String> getArgv(OS os, String pathString, String... args) {
    if (os != OS.WINDOWS || !isBatchScript(pathString)) {
      return ImmutableList.<String>builder().add(pathString).add(args).build();
    }

    // `cmd.exe` lives in C:\Windows\System32, which is always on the Windows search path.
    //
    // The entire command is wrapped in one extra pair of quotes: `/S` strips the first and the last
    // quote and executes what remains verbatim, which is the only reliable way to keep a program
    // path containing spaces in one piece.
    StringBuilder command = new StringBuilder("\"");
    command.append(quoteForCmd(pathString.replace('/', '\\')));
    for (String arg : args) {
      command.append(' ').append(quoteForCmd(arg));
    }
    command.append('"');

    return ImmutableList.of(
        "cmd.exe",
        "/S", // Strip the outer quotes and execute the rest as is.
        "/D", // Ignore AutoRun registry entries.
        "/C", // Execute the command that follows. Must be the last option.
        command.toString());
  }

  private static boolean isBatchScript(String path) {
    String lowerCasePath = Ascii.toLowerCase(path);
    return lowerCasePath.endsWith(".bat") || lowerCasePath.endsWith(".cmd");
  }

  /**
   * Quotes an argument for {@code cmd.exe}, but only when it would otherwise be split apart.
   *
   * <p>Quotes are left off whenever possible because a batch script sees them as part of {@code %1}
   * and friends.
   */
  private static String quoteForCmd(String arg) {
    return arg.isEmpty() || CharMatcher.whitespace().matchesAnyOf(arg) ? "\"" + arg + "\"" : arg;
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof CredentialHelper that) {
      return Objects.equals(this.getPath(), that.getPath());
    }

    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(getPath());
  }
}
