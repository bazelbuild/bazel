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

package com.google.devtools.build.lib.buildeventservice;

import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.proto.OptionFilters.OptionEffectTag;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.joda.time.Duration;

/** Options used by {@link BuildEventServiceModule}. */
public class BuildEventServiceOptions extends OptionsBase {

  @Option(
    name = "bes_backend",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Specifies the build event service (BES) backend endpoint as HOST or HOST:PORT. "
            + "Disabled by default."
  )
  public String besBackend;

  @Option(
    name = "bes_timeout",
    defaultValue = "0s",
    converter = DurationConverter.class,
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Specifies how long bazel should wait for the BES/BEP upload to complete after the "
            + "build and tests have finished. A valid timeout is a natural number followed by a "
            + "unit: Days (d), hours (h), minutes (m), seconds (s), and milliseconds (ms). The "
            + "default value is '0' which means that there is no timeout and that the upload will "
            + "continue in the background after a build has finished."
  )
  public Duration besTimeout;

  @Option(
    name = "bes_best_effort",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Specifies whether a failure to upload the BES protocol should also result in a build "
            + "failure. If 'false', bazel exits with ExitCode.PUBLISH_ERROR. (defaults to 'true')."
  )
  public boolean besBestEffort;

  @Option(
    name = "bes_lifecycle_events",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Specifies whether to publish BES lifecycle events. (defaults to 'true')."
  )
  public boolean besLifecycleEvents;

  @Option(
    name = "project_id",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Specifies the BES project identifier. Defaults to null."
  )
  public String projectId;

  /**
   * Simple String to Duration Converter.
   */
  public static class DurationConverter implements Converter<Duration> {

    private final Pattern durationRegex = Pattern.compile("^([0-9]+)(d|h|m|s|ms)$");

    @Override
    public Duration convert(String input) throws OptionsParsingException {
      // To be compatible with the previous parser, '0' doesn't need a unit.
      if ("0".equals(input)) {
        return Duration.ZERO;
      }
      Matcher m = durationRegex.matcher(input);
      if (!m.matches()) {
        throw new OptionsParsingException("Illegal duration '" + input + "'.");
      }
      long duration = Long.parseLong(m.group(1));
      String unit = m.group(2);
      switch(unit) {
        case "d":
          return Duration.standardDays(duration);
        case "h":
          return Duration.standardHours(duration);
        case "m":
          return Duration.standardMinutes(duration);
        case "s":
          return Duration.standardSeconds(duration);
        case "ms":
          return Duration.millis(duration);
        default:
          throw new IllegalStateException("This must not happen. Did you update the regex without "
              + "the switch case?");
      }
    }

    @Override
    public String getTypeDescription() {
      return "An immutable length of time.";
    }
  }

}
