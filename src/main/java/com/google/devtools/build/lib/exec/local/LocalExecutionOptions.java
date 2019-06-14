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
package com.google.devtools.build.lib.exec.local;

import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.RegexPatternOption;
import java.time.Duration;

/**
 * Local execution options.
 */
public class LocalExecutionOptions extends OptionsBase {

  @Option(
    name = "local_termination_grace_seconds",
    oldName = "local_sigkill_grace_seconds",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    defaultValue = "15",
    help =
        "Time to wait between terminating a local process due to timeout and forcefully "
            + "shutting it down."
  )
  public int localSigkillGraceSeconds;

  @Option(
      name = "allowed_local_actions_regex",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = Converters.RegexPatternConverter.class,
      defaultValue = "null",
      help =
          "A regex whitelist for action types which may be run locally. If unset, "
              + "all actions are allowed to execute locally")
  public RegexPatternOption allowedLocalAction;

  @Option(
    name = "experimental_collect_local_action_metrics",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.EXECUTION},
    help =
        "When enabled, execution statistics (such as user and system time) are recorded for "
            + "locally executed actions which don't use sandboxing"
  )
  public boolean collectLocalExecutionStatistics;

  public Duration getLocalSigkillGraceSeconds() {
    // TODO(ulfjack): Change localSigkillGraceSeconds type to Duration.
    return Duration.ofSeconds(localSigkillGraceSeconds);
  }
}
