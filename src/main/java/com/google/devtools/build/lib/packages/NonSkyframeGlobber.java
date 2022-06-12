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

/** {@link Globber} that uses {@link GlobCache} instead of Skyframe. */
public class NonSkyframeGlobber implements Globber {
  private final GlobCache globCache;

  NonSkyframeGlobber(GlobCache globCache) {
    this.globCache = globCache;
  }

  /** The {@link Globber.Token} used by {@link NonSkyframeGlobber}. */
  public static class Token extends Globber.Token {
    private final List<String> includes;
    private final List<String> excludes;
    private final Globber.Operation globberOperation;
    private final boolean allowEmpty;

    private Token(
        List<String> includes,
        List<String> excludes,
        Globber.Operation globberOperation,
        boolean allowEmpty) {
      this.includes = includes;
      this.excludes = excludes;
      this.globberOperation = globberOperation;
      this.allowEmpty = allowEmpty;
    }
  }

  @Override
  public Token runAsync(
      List<String> includes,
      List<String> excludes,
      Globber.Operation globberOperation,
      boolean allowEmpty)
      throws BadGlobException {

    for (String pattern : includes) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError = globCache.getGlobUnsortedAsync(pattern, globberOperation);
    }
    return new Token(includes, excludes, globberOperation, allowEmpty);
  }

  @Override
  public List<String> fetchUnsorted(Globber.Token token)
      throws BadGlobException, IOException, InterruptedException {
    Token ourToken = (Token) token;
    return globCache.globUnsorted(
        ourToken.includes, ourToken.excludes, ourToken.globberOperation, ourToken.allowEmpty);
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
