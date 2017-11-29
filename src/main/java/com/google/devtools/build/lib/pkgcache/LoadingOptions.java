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
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;
import java.util.Set;

/**
 * Options that affect how command-line target patterns are resolved to individual targets.
 */
public class LoadingOptions extends OptionsBase {
  @Option(
    name = "build_tests_only",
    defaultValue = "false",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If specified, only *_test and test_suite rules will be built and other targets specified "
            + "on the command line will be ignored. By default everything that was requested "
            + "will be built."
  )
  public boolean buildTestsOnly;

  @Option(
    name = "compile_one_dependency",
    defaultValue = "false",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Compile a single dependency of the argument files. This is useful for syntax checking "
            + "source files in IDEs, for example, by rebuilding a single target that depends on "
            + "the source file to detect errors as early as possible in the edit/build/test cycle. "
            + "This argument affects the way all non-flag arguments are interpreted; instead of "
            + "being targets to build they are source filenames.  For each source filename "
            + "an arbitrary target that depends on it will be built."
  )
  public boolean compileOneDependency;

  @Option(
    name = "build_tag_filters",
    converter = CommaSeparatedOptionListConverter.class,
    defaultValue = "",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies a comma-separated list of tags. Each tag can be optionally "
            + "preceded with '-' to specify excluded tags. Only those targets will be built that "
            + "contain at least one included tag and do not contain any excluded tags. This option "
            + "does not affect the set of tests executed with the 'test' command; those are be "
            + "governed by the test filtering options, for example '--test_tag_filters'"
  )
  public List<String> buildTagFilterList;

  @Option(
    name = "test_tag_filters",
    converter = CommaSeparatedOptionListConverter.class,
    defaultValue = "",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies a comma-separated list of test tags. Each tag can be optionally "
            + "preceded with '-' to specify excluded tags. Only those test targets will be "
            + "found that contain at least one included tag and do not contain any excluded "
            + "tags. This option affects --build_tests_only behavior and the test command."
  )
  public List<String> testTagFilterList;

  @Option(
    name = "test_size_filters",
    converter = TestSize.TestSizeFilterConverter.class,
    defaultValue = "",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies a comma-separated list of test sizes. Each size can be optionally "
            + "preceded with '-' to specify excluded sizes. Only those test targets will be "
            + "found that contain at least one included size and do not contain any excluded "
            + "sizes. This option affects --build_tests_only behavior and the test command."
  )
  public Set<TestSize> testSizeFilterSet;

  @Option(
    name = "test_timeout_filters",
    converter = TestTimeout.TestTimeoutFilterConverter.class,
    defaultValue = "",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies a comma-separated list of test timeouts. Each timeout can be "
            + "optionally preceded with '-' to specify excluded timeouts. Only those test "
            + "targets will be found that contain at least one included timeout and do not "
            + "contain any excluded timeouts. This option affects --build_tests_only behavior "
            + "and the test command."
  )
  public Set<TestTimeout> testTimeoutFilterSet;

  @Option(
    name = "test_lang_filters",
    converter = CommaSeparatedOptionListConverter.class,
    defaultValue = "",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies a comma-separated list of test languages. Each language can be "
            + "optionally preceded with '-' to specify excluded languages. Only those "
            + "test targets will be found that are written in the specified languages. "
            + "The name used for each language should be the same as the language prefix in the "
            + "*_test rule, e.g. one of 'cc', 'java', 'py', etc. "
            + "This option affects --build_tests_only behavior and the test command."
  )
  public List<String> testLangFilterList;

  @Option(
    name = "build_manual_tests",
    defaultValue = "false",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Forces test targets tagged 'manual' to be built. 'manual' tests are excluded from "
            + "processing. This option forces them to be built (but not executed)."
  )
  public boolean buildManualTests;

  // If this option is set, the value of experimental_interleave_loading_and_analysis is completely
  // ignored. This enables a different LoadingPhaseRunner implementation which doesn't implement
  // the loading phase at all, and therefore can't currently support the other flag. If we roll this
  // out soonish, then we're never going to implement the legacy code path in the new
  // implementation, making it a moot point.
  @Option(
    name = "experimental_skyframe_target_pattern_evaluator",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Use the Skyframe-based target pattern evaluator; implies "
            + "--experimental_interleave_loading_and_analysis."
  )
  public boolean useSkyframeTargetPatternEvaluator;
}
