// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.metrics;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

/**
 * A blaze module that installs metrics instrumentations and issues a {@link BuildMetricsEvent} at
 * the end of the build.
 */
public class MetricsModule extends BlazeModule {

  /** Metrics options. */
  public static final class Options extends OptionsBase {
    @Option(
        name = "bep_publish_used_heap_size_post_build",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.LOGGING,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "When set we collect and publish used_heap_size_post_build "
                + "from build_event_stream.proto. This forces a full GC and is off by default.")
    public boolean bepPublishUsedHeapSizePostBuild;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name()) ? ImmutableList.of(Options.class) : ImmutableList.of();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    MetricsCollector.installInEnv(env);
  }

  @Override
  public void afterCommand() {}
}
