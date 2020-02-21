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

package com.google.devtools.build.lib.packages;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.Future;

/** {@link Globber} that uses the legacy GlobCache. */
public class LegacyGlobber implements Globber {
  private final GlobCache globCache;

  LegacyGlobber(GlobCache globCache) {
    this.globCache = globCache;
  }

  private static class Token extends Globber.Token {
    public final List<String> includes;
    public final List<String> excludes;
    public final boolean excludeDirs;
    public final boolean allowEmpty;

    public Token(
        List<String> includes, List<String> excludes, boolean excludeDirs, boolean allowEmpty) {
      this.includes = includes;
      this.excludes = excludes;
      this.excludeDirs = excludeDirs;
      this.allowEmpty = allowEmpty;
    }
  }

  @Override
  public Token runAsync(
      List<String> includes, List<String> excludes, boolean excludeDirs, boolean allowEmpty)
      throws BadGlobException {
    for (String pattern : includes) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError = globCache.getGlobUnsortedAsync(pattern, excludeDirs);
    }
    return new Token(includes, excludes, excludeDirs, allowEmpty);
  }

  @Override
  public List<String> fetchUnsorted(Globber.Token token)
      throws BadGlobException, IOException, InterruptedException {
    Token legacyToken = (Token) token;
    return globCache.globUnsorted(
        legacyToken.includes,
        legacyToken.excludes,
        legacyToken.excludeDirs,
        legacyToken.allowEmpty);
  }

  @Override
  public void onInterrupt() {
    globCache.cancelBackgroundTasks();
  }

  @Override
  public void onCompletion() {
    globCache.finishBackgroundTasks();
  }
}
