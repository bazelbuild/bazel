// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Utf8;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions.Utf8EnforcementMode;
import java.util.Optional;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.ParserInput;

/** Helper functions for Skyframe. */
public final class SkyframeUtil {

  /**
   * Produces a {@link ParserInput} from the raw bytes of a file while optionally enforcing that the
   * contents are valid UTF-8.
   *
   * <p><b>Warnings and errors are reported to the {@link EventHandler}.</b>
   *
   * @return an optional with a {@link ParserInput} if the bytes are valid UTF-8 or the enforcement
   *     mode is not {@link Utf8EnforcementMode#ERROR}, or an empty {@link Optional} otherwise.
   */
  // This method is the only one that is supposed to use the deprecated ParserInput.fromLatin1
  // method.
  @SuppressWarnings("deprecation")
  public static Optional<ParserInput> createParserInput(
      byte[] bytes, String file, Utf8EnforcementMode utf8EnforcementMode, EventHandler reporter) {
    switch (utf8EnforcementMode) {
      case OFF -> {}
      case WARNING -> {
        if (!Utf8.isWellFormed(bytes)) {
          reporter.handle(
              Event.warn(
                  Location.fromFile(file),
                  "not a valid UTF-8 encoded file; this can lead to inconsistent behavior and"
                      + " will be disallowed in a future version of Bazel"));
        }
      }
      case ERROR -> {
        if (!Utf8.isWellFormed(bytes)) {
          reporter.handle(
              Event.error(
                  Location.fromFile(file),
                  "not a valid UTF-8 encoded file; this can lead to inconsistent behavior and"
                      + " will be disallowed in a future version of Bazel"));
          return Optional.empty();
        }
      }
    }
    return Optional.of(ParserInput.fromLatin1(bytes, file));
  }

  private SkyframeUtil() {}
}
