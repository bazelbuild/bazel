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

import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Utils for logging safely user commandlines. */
public class SafeRequestLogging {
  private static final Pattern suppressFromLog =
      Pattern.compile(
          "--client_env=([^=]*(?:auth|pass|cookie|token)[^=]*)=", Pattern.CASE_INSENSITIVE);

  private SafeRequestLogging() {}

  /**
   * Generates a string form of a request to be written to the logs, filtering the user environment
   * to remove anything that looks private. The current filter criteria removes any variable whose
   * name includes "auth", "pass", "cookie" or "token".
   *
   * @return the filtered request to write to the log.
   */
  public static String getRequestLogString(List<String> requestStrings) {
    StringBuilder buf = new StringBuilder();
    buf.append('[');
    String sep = "";
    Matcher m = suppressFromLog.matcher("");
    for (String s : requestStrings) {
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
