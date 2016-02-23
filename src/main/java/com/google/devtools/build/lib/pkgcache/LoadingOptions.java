// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;

import java.util.List;
import java.util.Set;

/**
 * Options that affect how command-line target patterns are resolved to individual targets.
 */
public class LoadingOptions extends OptionsBase {

  @Option(name = "loading_phase_threads",
      defaultValue = "-1",
      category = "undocumented",
      converter = LoadingPhaseThreadCountConverter.class,
      help = "Number of parallel threads to use for the loading phase.")
  public int loadingPhaseThreads;

  @Option(name = "build_tests_only",
      defaultValue = "false",
      category = "what",
      help = "If specified, only *_test and test_suite rules will be built "
        + "and other targets specified on the command line will be ignored. "
        + "By default everything that was requested will be built.")
  public boolean buildTestsOnly;

  @Option(name = "compile_one_dependency",
          defaultValue = "false",
          category = "what",
          help = "Compile a single dependency of the argument files.  "
          + "This is useful for syntax checking source files in IDEs, "
          + "for example, by rebuilding a single target that depends on "
          + "the source file to detect errors as early as possible in the "
          + "edit/build/test cycle.  This argument affects the way all "
          + "non-flag arguments are interpreted; instead of being targets "
          + "to build they are source filenames.  For each source filename "
          + "an arbitrary target that depends on it will be built.")
  public boolean compileOneDependency;

  @Option(name = "test_tag_filters",
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "",
      category = "what",
      help = "Specifies a comma-separated list of test tags. Each tag can be optionally " +
             "preceded with '-' to specify excluded tags. Only those test targets will be " +
             "found that contain at least one included tag and do not contain any excluded " +
             "tags. This option affects --build_tests_only behavior and the test command."
      )
  public List<String> testTagFilterList;

  @Option(name = "test_size_filters",
      converter = TestSize.TestSizeFilterConverter.class,
      defaultValue = "",
      category = "what",
      help = "Specifies a comma-separated list of test sizes. Each size can be optionally " +
             "preceded with '-' to specify excluded sizes. Only those test targets will be " +
             "found that contain at least one included size and do not contain any excluded " +
             "sizes. This option affects --build_tests_only behavior and the test command."
      )
  public Set<TestSize> testSizeFilterSet;

  @Option(name = "test_timeout_filters",
      converter = TestTimeout.TestTimeoutFilterConverter.class,
      defaultValue = "",
      category = "what",
      help = "Specifies a comma-separated list of test timeouts. Each timeout can be " +
             "optionally preceded with '-' to specify excluded timeouts. Only those test " +
             "targets will be found that contain at least one included timeout and do not " +
             "contain any excluded timeouts. This option affects --build_tests_only behavior " +
             "and the test command."
      )
  public Set<TestTimeout> testTimeoutFilterSet;

  @Option(name = "test_lang_filters",
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "",
      category = "what",
      help = "Specifies a comma-separated list of test languages. Each language can be " +
             "optionally preceded with '-' to specify excluded languages. Only those " +
             "test targets will be found that are written in the specified languages. " +
             "The name used for each language should be the same as the language prefix in the " +
             "*_test rule, e.g. one of 'cc', 'java', 'py', etc." +
             "This option affects --build_tests_only behavior and the test command."
      )
  public List<String> testLangFilterList;

  // If this option is set, the value of experimental_interleave_loading_and_analysis is completely
  // ignored. This enables a different LoadingPhaseRunner implementation which doesn't implement
  // the loading phase at all, and therefore can't currently support the other flag. If we roll this
  // out soonish, then we're never going to implement the legacy code path in the new
  // implementation, making it a moot point.
  @Option(name = "experimental_skyframe_target_pattern_evaluator",
      defaultValue = "false",
      category = "hidden",
      help = "Use the Skyframe-based target pattern evaluator; implies "
          + "--experimental_interleave_loading_and_analysis.")
  public boolean useSkyframeTargetPatternEvaluator;

  /**
   * A converter for loading phase thread count. Since the default is not a true constant, we
   * create a converter here to implement the default logic.
   */
  public static final class LoadingPhaseThreadCountConverter implements Converter<Integer> {
    @Override
    public Integer convert(String input) throws OptionsParsingException {
      if ("-1".equals(input)) {
        // Reduce thread count while running tests. Test cases are typically small, and large thread
        // pools vying for a relatively small number of CPU cores may induce non-optimal
        // performance.
        return System.getenv("TEST_TMPDIR") == null ? 200 : 5;
      }

      try {
        return Integer.decode(input);
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
