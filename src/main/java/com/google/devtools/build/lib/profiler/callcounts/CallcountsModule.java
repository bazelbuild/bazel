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

package com.google.devtools.build.lib.profiler.callcounts;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;

/**
 * Developer utility. Add calls to {@link Callcounts#registerCall} in places that you want to count
 * call occurrences with call stacks. At the end of the command, a pprof-compatible file will be
 * dumped to the path specified by --call_count_output_path.
 */
public class CallcountsModule extends BlazeModule {
  private static final int MAX_CALLSTACK = 8;
  // Every Nth call is actually logged
  private static final int SAMPLE_PERIOD = 100;
  // We use some variance to avoid patterns where the same calls get logged over and over
  private static final int SAMPLE_VARIANCE = 10;

  private String outputPath;
  private Reporter reporter;

  /** Options for {@link CallcountsModule}. */
  public static class CallcountsOptions extends OptionsBase {
    @Option(
      name = "call_count_output_path",
      help =
          "An absolute path to a pprof gz file. "
              + "At the end of the command a call count file will be written.",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED, // Devs only
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = ""
    )
    public String outputPath;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of(CallcountsOptions.class);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    this.outputPath = env.getOptions().getOptions(CallcountsOptions.class).outputPath;
    Callcounts.init(SAMPLE_PERIOD, SAMPLE_VARIANCE, MAX_CALLSTACK);
    this.reporter = env.getReporter();
  }

  @Override
  public void afterCommand() {
    if (outputPath != null && !outputPath.isEmpty()) {
      try {
        Callcounts.dump(outputPath);
      } catch (IOException e) {
        reporter.error(null, "Error saving callcount file", e);
      }
    }
    Callcounts.reset();
    this.reporter = null;
    this.outputPath = null;
  }
}
