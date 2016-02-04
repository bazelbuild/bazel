// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.util.Pair;

import java.io.IOException;
import java.util.List;
import java.util.Set;

/** Interface for evaluating globs during package loading. */
public interface Globber {
  /** An opaque token for fetching the result of a glob computation. */
  abstract static class Token {}

  /** Used to indicate an invalid glob pattern. */
  static class BadGlobException extends Exception {
    public BadGlobException(String message) {
      super(message);
    }
  }

  /**
   * Asynchronously starts the given glob computation and returns a token for fetching the
   * result.
   *
   * @throws BadGlobException if any of the patterns in {@code includes} or {@code excludes} are
   *     invalid.
   */
  Token runAsync(List<String> includes, List<String> excludes, boolean excludeDirs)
      throws BadGlobException;

  /** Fetches the result of a previously started glob computation. */
  List<String> fetch(Token token) throws IOException, InterruptedException;

  /** Should be called when the globber is about to be discarded due to an interrupt. */
  void onInterrupt();

  /** Should be called when the globber is no longer needed. */
  void onCompletion();

  /** Returns all the glob computations requested before {@link #onCompletion} was called. */
    Set<Pair<String, Boolean>> getGlobPatterns();
}
