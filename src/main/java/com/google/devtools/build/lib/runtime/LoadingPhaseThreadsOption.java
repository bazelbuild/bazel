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

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.util.ResourceConverter;
import com.google.devtools.build.lib.util.TestType;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;

/** Defines the --loading_phase_threads option which is used by multiple commands. */
public class LoadingPhaseThreadsOption extends OptionsBase {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  @Option(
      name = "loading_phase_threads",
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = LoadingPhaseThreadCountConverter.class,
      help =
          "Number of parallel threads to use for the loading/analysis phase."
              + "Takes "
              + ResourceConverter.FLAG_SYNTAX
              + ". \"auto\" sets a reasonable default based on "
              + "host resources. Must be at least 1.")
  public int threads;

  /**
   * A converter for loading phase thread count. Takes {@value FLAG_SYNTAX}. Caps at 20 for tests.
   */
  public static final class LoadingPhaseThreadCountConverter extends ResourceConverter {

    public LoadingPhaseThreadCountConverter() {
      // TODO(jmmv): Using the number of cores has proven to yield reasonable analysis times on
      // Mac Pros and MacBook Pros but we should probably do better than this. (We haven't made
      // any guarantees that "auto" means number of cores precisely to leave us room to tune this
      // further in the future.)
      super(() -> (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getCpuUsage()));
    }

    @Override
    public int checkAndLimit(int value) throws OptionsParsingException {
      // Cap thread count while running tests. Test cases are typically small and large thread
      // pools vying for a relatively small number of CPU cores may induce non-optimal
      // performance.
      //
      // TODO(jmmv): If tests care about this, it's them who should be setting a cap.
      if (TestType.isInTest()) {
        value = Math.min(20, value);
        logger.atInfo().log("Running under a test; loading_phase_threads capped at %d", value);
      }
      return super.checkAndLimit(value);
    }
  }
}
