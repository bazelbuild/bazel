// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;

/** Defines the --loading_phase_threads option which is used by multiple commands. */
public class LoadingPhaseThreadsOption extends OptionsBase {
  @Option(
    name = "loading_phase_threads",
    defaultValue = LoadingPhaseThreadCountConverter.DEFAULT_VALUE,
    documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    converter = LoadingPhaseThreadCountConverter.class,
    help = "Number of parallel threads to use for the loading/analysis phase."
  )
  public int threads;

  /**
   * A converter for loading phase thread count. Since the default is not a true constant, we create
   * a converter here to implement the default logic.
   */
  public static final class LoadingPhaseThreadCountConverter implements Converter<Integer> {

    private static final String DEFAULT_VALUE = "default";
    // Reduce thread count while running tests. Test cases are typically small, and large thread
    // pools vying for a relatively small number of CPU cores may induce non-optimal
    // performance.
    private static final int NON_TEST_THREADS = 200;
    private static final int TEST_THREADS = 20;

    @Override
    public Integer convert(String input) throws OptionsParsingException {
      if (DEFAULT_VALUE.equals(input)) {
        return System.getenv("TEST_TMPDIR") == null ? NON_TEST_THREADS : TEST_THREADS;
      }

      try {
        int result = Integer.decode(input);
        if (result < 1) {
          throw new OptionsParsingException("'" + input + "' must be at least 1");
        }
        return result;
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not an int");
      }
    }

    @Override
    public String getTypeDescription() {
      return "an integer";
    }
  }
}
