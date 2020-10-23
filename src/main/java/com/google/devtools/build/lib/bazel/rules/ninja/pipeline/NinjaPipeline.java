// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.pipeline;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaFileParseResult;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaFileParseResult.NinjaPromise;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaVariableValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;

/**
 * Interface Responsible for parsing Ninja file, all its included and subninja files, and returning
 * {@link NinjaScope} with rules and expanded variables, and list of {@link NinjaTarget}.
 *
 * <p>This interface exists to break the package cycle between parser and pipeline.
 */
public interface NinjaPipeline {

  /**
   * Synchronously or asynchronously schedules parsing of the included Ninja file, and returns
   * {@link NinjaPromise<NinjaFileParseResult>} - an object, from which we can obtain the result of
   * the parsing - {@link NinjaFileParseResult} - after we resolved variables in the parent file to
   * the point when the child file was included.
   *
   * <p>In some cases, include and subninja statements can contain variable references, then we must
   * postpone scheduling the parsing of the file until variable is expanded and the path to file
   * becomes known.
   *
   * <p>That is why we use {@link NinjaPromise<NinjaFileParseResult>} to save the future file
   * parsing result in the parent file {@link NinjaFileParseResult} structure.
   */
  NinjaPromise<NinjaFileParseResult> createChildFileParsingPromise(
      NinjaVariableValue value, long offset, String parentNinjaFileName)
      throws GenericParsingException, IOException;

  /**
   * An interner for {@link PathFragment} instances for the inputs and outputs of {@link
   * NinjaTarget}.
   */
  Interner<PathFragment> getPathFragmentInterner();

  /** An String interner for rule and build statements' variable names. */
  Interner<String> getNameInterner();
}

