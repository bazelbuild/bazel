// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.Set;

/**
 * A Preprocessor is an interface to implement generic text-based
 * preprocessing of BUILD files.
 */
public interface Preprocessor {
  /**
   * Creates Preprocessor instances.
   */
  public interface Factory {
    /**
     * Returns a Preprocessor instance. The CachingPackageLocator is
     * provided so the preprocessor can lookup other BUILD files.
     * Currently, newPreprocessor() is called every time a BUILD
     * preprocessor parse is attempted.
     */
    Preprocessor newPreprocessor(CachingPackageLocator loc);
  }

  /**
   * A (result, success) tuple indicating the outcome of preprocessing.
   */
  public static class Result {
    public final ParserInputSource result;
    public final boolean preprocessed;
    public final boolean containsTransientErrors;

    public Result(ParserInputSource result,
        boolean preprocessed, boolean containsTransientErrors) {
      this.result = result;
      this.preprocessed = preprocessed;
      this.containsTransientErrors = containsTransientErrors;
    }

    public static Result success(ParserInputSource result, boolean preprocessed) {
      return new Result(result, preprocessed, false);
    }

    // This error is used only if the BUILD file is not in Python syntax.
    public static Result preprocessingError(Path buildFile) {
      return new Result(ParserInputSource.create("", buildFile), true, false);
    }

    // Signals some other preprocessing error.
    public static Result transientError(Path buildFile) {
      return new Result(ParserInputSource.create("", buildFile), false, true);
    }
  }

  /**
   * Returns a Result resulting from applying Python preprocessing to the contents of "in". If
   * errors happen, they must be reported both as an event on listener and in the function
   * return value.
   *
   * @param in the BUILD file to be preprocessed.
   * @param packageName the BUILD file's package.
   * @param globCache
   * @param listener a listener on which to report warnings/errors.
   * @param globalEnv the GLOBALS Python environment.
   * @param ruleNames the set of names of all rules in the build language.
   * @throws IOException if there was an I/O problem during preprocessing.
   * @return a pair of the ParserInputSource and a map of subincludes seen during the evaluation
   */
  public Result preprocess(
      ParserInputSource in,
      String packageName,
      GlobCache globCache,
      ErrorEventListener listener,
      Environment globalEnv,
      Set<String> ruleNames)
    throws IOException, InterruptedException;
}
