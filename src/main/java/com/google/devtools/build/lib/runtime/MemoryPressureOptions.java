// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.time.Duration;

/** Options for responding to memory pressure. */
public final class MemoryPressureOptions extends OptionsBase {

  @Option(
      name = "experimental_oom_more_eagerly_threshold",
      defaultValue = "100",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      help =
          "If this flag is set to a value less than 100, Bazel will OOM if, after two full GC's, "
              + "more than this percentage of the (old gen) heap is still occupied.")
  public int oomMoreEagerlyThreshold;

  @Option(
      name = "min_time_between_triggered_gc",
      defaultValue = "1m",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "The minimum amount of time between GCs triggered by RetainedHeapLimiter.")
  public Duration minTimeBetweenTriggeredGc;

  @Option(
      name = "skyframe_high_water_mark_threshold",
      defaultValue = "85",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      help =
          "Flag for advanced configuration of Bazel's internal Skyframe engine. If Bazel detects"
              + " its retained heap percentage usage is at least this threshold, it will drop"
              + " unnecessary temporary Skyframe state. Tweaking this may let you mitigate wall"
              + " time impact of GC thrashing, when the GC thrashing is (i) caused by the memory"
              + " usage of this temporary state and (ii) more costly than reconstituting the state"
              + " when it is needed.")
  public int skyframeHighWaterMarkMemoryThreshold;

  @Option(
      name = "skyframe_high_water_mark_minor_gc_drops_per_invocation",
      defaultValue = "" + Integer.MAX_VALUE,
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      converter = NonNegativeIntegerConverter.class,
      help =
          "Flag for advanced configuration of Bazel's internal Skyframe engine. If Bazel detects"
              + " its retained heap percentage usage exceeds the threshold set by"
              + " --skyframe_high_water_mark_threshold, when a minor GC event occurs, it will drop"
              + " unnecessary temporary Skyframe state, up to this many times per invocation."
              + " Defaults to Integer.MAX_VALUE; effectively unlimited. Zero means that minor GC"
              + " events will never trigger drops. If the limit is reached, Skyframe state will no"
              + " longer be dropped when a minor GC event occurs and that retained heap percentage"
              + " threshold is exceeded.")
  public int skyframeHighWaterMarkMinorGcDropsPerInvocation;

  @Option(
      name = "skyframe_high_water_mark_full_gc_drops_per_invocation",
      defaultValue = "" + Integer.MAX_VALUE,
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      converter = NonNegativeIntegerConverter.class,
      help =
          "Flag for advanced configuration of Bazel's internal Skyframe engine. If Bazel detects"
              + " its retained heap percentage usage exceeds the threshold set by"
              + " --skyframe_high_water_mark_threshold, when a full GC event occurs, it will drop"
              + " unnecessary temporary Skyframe state, up to this many times per invocation."
              + " Defaults to Integer.MAX_VALUE; effectively unlimited. Zero means that full GC"
              + " events will never trigger drops. If the limit is reached, Skyframe state will no"
              + " longer be dropped when a full GC event occurs and that retained heap percentage"
              + " threshold is exceeded.")
  public int skyframeHighWaterMarkFullGcDropsPerInvocation;

  static class NonNegativeIntegerConverter extends Converters.RangeConverter {
    public NonNegativeIntegerConverter() {
      super(0, Integer.MAX_VALUE);
    }
  }
}
