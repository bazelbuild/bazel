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

import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.logging.Logger;

/** Defines the --loading_phase_threads option which is used by multiple commands. */
public class LoadingPhaseThreadsOption extends OptionsBase {

  private static final Logger logger = Logger.getLogger(LoadingPhaseThreadsOption.class.getName());

  @Option(
    name = "loading_phase_threads",
    defaultValue = "auto",
    documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    converter = LoadingPhaseThreadCountConverter.class,
    help = "Number of parallel threads to use for the loading/analysis phase."
        + " \"auto\" means to use a reasonable value derived from the machine's hardware profile"
        + " (e.g. the number of processors)."
  )
  public int threads;

  /**
   * A converter for loading phase thread count. Since the default is not a true constant, we create
   * a converter here to implement the default logic.
   */
  public static final class LoadingPhaseThreadCountConverter implements Converter<Integer> {

    @Override
    public Integer convert(String input) throws OptionsParsingException {
      int threads;
      if (input.equals("auto")) {
        threads = (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getCpuUsage());
        logger.info("Flag \"loading_phase_threads\" was set to \"auto\"; using " + threads
            + " threads");
      } else {
        try {
          threads = Integer.decode(input);
          if (threads < 1) {
            throw new OptionsParsingException("'" + input + "' must be at least 1");
          }
        } catch (NumberFormatException e) {
          throw new OptionsParsingException("'" + input + "' is not an int");
        }
      }

      if (System.getenv("TEST_TMPDIR") != null) {
        // Cap thread count while running tests. Test cases are typically small and large thread
        // pools vying for a relatively small number of CPU cores may induce non-optimal
        // performance.
        //
        // TODO(jmmv): If tests care about this, it's them who should be setting a cap.
        threads = Math.min(20, threads);
        logger.info("Running under a test; loading_phase_threads capped at " + threads);
      }

      // TODO(jmmv): Using the number of cores has proven to yield reasonable analysis times on
      // Mac Pros and MacBook Pros but we should probably do better than this. (We haven't made
      // any guarantees that "auto" means number of cores precisely to leave us room to tune this
      // further in the future.)
      return threads;
    }

    @Override
    public String getTypeDescription() {
      return "\"auto\" or an integer";
    }
  }
}
