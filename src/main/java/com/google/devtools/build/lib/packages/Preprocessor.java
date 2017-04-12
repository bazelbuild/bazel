// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Set;

/** A Preprocessor is an interface to implement generic text-based preprocessing of BUILD files. */
public interface Preprocessor {
  /**
   * A (result, success) tuple indicating the outcome of preprocessing.
   */
  static class Result {
    public final ParserInputSource result;

    private Result(ParserInputSource result) {
      this.result = result;
    }

    public static Result noPreprocessing(PathFragment buildFilePathFragment,
        byte[] buildFileBytes) {
      return noPreprocessing(ParserInputSource.create(
          FileSystemUtils.convertFromLatin1(buildFileBytes), buildFilePathFragment));
    }

    /** Convenience factory for a {@link Result} wrapping non-preprocessed BUILD file contents. */
    public static Result noPreprocessing(ParserInputSource buildFileSource) {
      return new Result(buildFileSource);
    }
  }

  /**
   * Returns a Result resulting from applying Python preprocessing to the contents of "in". If
   * errors happen, they must be reported both as an event on eventHandler and in the function
   * return value. An IOException is only thrown when preparing for preprocessing. Once
   * preprocessing actually begins, any I/O problems encountered will be reflected in the return
   * value, not manifested as exceptions.
   *
   * @param buildFilePath the BUILD file to be preprocessed.
   * @param buildFileBytes the raw contents of the BUILD file to be preprocessed.
   * @param packageName the BUILD file's package.
   * @param globber a globber for evaluating globs.
   * @param globals the global bindings for the Python environment.
   * @param ruleNames the set of names of all rules in the build language.
   * @throws IOException if there was an I/O problem preparing for preprocessing.
   * @return a pair of the ParserInputSource and a map of subincludes seen during the evaluation
   */
  Result preprocess(
      Path buildFilePath,
      byte[] buildFileBytes,
      String packageName,
      Globber globber,
      Set<String> ruleNames)
    throws IOException, InterruptedException;

  /** The result of parsing a preprocessed BUILD file. */
  static class AstAfterPreprocessing {
    public final BuildFileAST ast;
    public final boolean containsAstParsingErrors;
    public final Iterable<Event> allEvents;

    public AstAfterPreprocessing(Result preprocessingResult, BuildFileAST ast,
        StoredEventHandler astParsingEventHandler) {
      this.ast = ast;
      this.containsAstParsingErrors = astParsingEventHandler.hasErrors();
      this.allEvents = astParsingEventHandler.getEvents();
    }
  }
}
