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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;

/** Flags specific to test summary reporting. */
public class TestSummaryOptions extends OptionsBase {
  public static final TestSummaryOptions DEFAULTS = Options.getDefaults(TestSummaryOptions.class);

  @Option(
      name = "verbose_test_summary",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, print additional information (timing, number of failed runs, etc) in the"
              + " test summary.")
  public boolean verboseSummary;

  @Option(
      name = "test_verbose_timeout_warnings",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, print additional warnings when the actual test execution time does not "
              + "match the timeout defined by the test (whether implied or explicit).")
  public boolean testVerboseTimeoutWarnings;

  @Option(
      name = "print_relative_test_log_paths",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, when printing the path to a test log, use relative path that makes use of "
              + "the 'testlogs' convenience symlink. N.B. - A subsequent 'build'/'test'/etc "
              + "invocation with a different configuration can cause the target of this symlink "
              + "to change, making the path printed previously no longer useful.")
  public boolean printRelativeTestLogPaths;
}
