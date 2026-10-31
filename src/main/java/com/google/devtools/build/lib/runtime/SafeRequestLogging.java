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

package com.google.devtools.build.lib.runtime;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Utils for logging safely user commandlines. */
public class SafeRequestLogging {
  /**
   * Regex fragment matching a client environment variable name that looks private. Variable names
   * cannot contain '=', so {@code [^=]*} is bounded to the name.
   */
  private static final String PRIVATE_LOOKING_ENV_NAME =
      "[^=]*(?:auth|pass|cookie|token|api_key|credential|secret)[^=]*";

  /**
   * Matches {@code --client_env=NAME=} for names that should be redacted. Replaced wholesale by
   * {@link #setRedactedClientEnvNames} when the user supplies additional names via {@code
   * --redact_client_env}.
   */
  private static volatile Pattern suppressFromLog = buildClientEnvPattern(ImmutableList.of());

  /**
   * Installs the list of additional client environment variable names to redact, as given by the
   * {@code --redact_client_env} startup option. Each name matches the whole variable name,
   * case-insensitively; {@code *} matches any run of characters and everything else is literal.
   * An empty list restores the built-in heuristic alone.
   */
  static void setRedactedClientEnvNames(List<String> names) {
    suppressFromLog = buildClientEnvPattern(names);
  }

  /** Builds the {@code --client_env} redaction pattern: built-in heuristic plus {@code names}. */
  static Pattern buildClientEnvPattern(List<String> names) {
    StringBuilder regex = new StringBuilder("--client_env=(").append(PRIVATE_LOOKING_ENV_NAME);
    for (String name : names) {
      regex.append('|').append(globToRegex(name));
    }
    return Pattern.compile(regex.append(")=").toString(), Pattern.CASE_INSENSITIVE);
  }

  /** Translates a name where only {@code *} is special into a regex bounded to a variable name. */
  private static String globToRegex(String name) {
    return Arrays.stream(name.split("\\*", -1)).map(Pattern::quote).collect(joining("[^=]*"));
  }

  private static final ImmutableSet<String> CREDENTIAL_OPTION_NAMES =
      ImmutableSet.of(
          "--invocation_policy",
          "--bes_header",
          "--remote_header",
          "--remote_cache_header",
          "--remote_exec_header",
          "--remote_downloader_header",
          "--tls_client_key",
          "--google_credentials");

  private static final Pattern CREDENTIAL_OPTIONS =
      Pattern.compile(
          "--("
              + CREDENTIAL_OPTION_NAMES.stream().map(s -> s.substring(2)).collect(joining("|"))
              + ")=",
          Pattern.CASE_INSENSITIVE);

  public static ImmutableList<String> redactArguments(List<String> args) {
    ImmutableList.Builder<String> redactedArgs = ImmutableList.builder();
    Matcher credMatcher = CREDENTIAL_OPTIONS.matcher("");
    boolean redactNext = false;
    for (String arg : args) {
      if (redactNext) {
        redactedArgs.add("<REDACTED>");
        redactNext = false;
        continue;
      }
      credMatcher.reset(arg);
      if (credMatcher.lookingAt()) {
        redactedArgs.add(credMatcher.group() + "<REDACTED>");
      } else if (CREDENTIAL_OPTION_NAMES.contains(arg)) {
        redactedArgs.add(arg);
        redactNext = true;
      } else {
        redactedArgs.add(arg);
      }
    }
    return redactedArgs.build();
  }

  private SafeRequestLogging() {}

  /**
   * Generates a string form of a request to be written to the logs, filtering the user environment
   * to remove anything that looks private. The current filter criteria removes any variable whose
   * name includes "auth", "pass", "cookie", "token", "api_key", "credential" or "secret", plus any
   * variable named by the {@code --redact_client_env} startup option.
   *
   * @return the filtered request to write to the log.
   */
  public static String getRequestLogString(List<String> requestStrings) {
    StringBuilder buf = new StringBuilder();
    buf.append('[');
    String sep = "";
    Matcher m = suppressFromLog.matcher("");
    for (String s : redactArguments(requestStrings)) {
      buf.append(sep);
      m.reset(s);
      if (m.lookingAt()) {
        buf.append(m.group());
        buf.append("__private_value_removed__");
      } else {
        buf.append(s);
      }
      sep = ", ";
    }
    buf.append(']');
    return buf.toString();
  }
}
