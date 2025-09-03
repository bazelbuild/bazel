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

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Converters.DurationConverter;
import com.google.devtools.common.options.Converters.OptionalPercentageConverter;
import com.google.devtools.common.options.Converters.PercentageConverter;
import com.google.devtools.common.options.Converters.RangeConverter;
import com.google.devtools.common.options.Converters.RegexPatternConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.RegexPatternOption;
import java.time.Duration;
import java.util.OptionalInt;

/** Options for responding to memory pressure. */
public final class MemoryPressureOptions extends OptionsBase {

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
      defaultValue = "10",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      converter = NonNegativeIntegerConverter.class,
      help =
          "Flag for advanced configuration of Bazel's internal Skyframe engine. If Bazel detects"
              + " its retained heap percentage usage exceeds the threshold set by"
              + " --skyframe_high_water_mark_threshold, when a minor GC event occurs, it will drop"
              + " unnecessary temporary Skyframe state, up to this many times per invocation."
              + " Defaults to 10. Zero means that minor GC events will never trigger drops. If the"
              + " limit is reached, Skyframe state will no longer be dropped when a minor GC event"
              + " occurs and that retained heap percentage threshold is exceeded.")
  public int skyframeHighWaterMarkMinorGcDropsPerInvocation;

  @Option(
      name = "skyframe_high_water_mark_full_gc_drops_per_invocation",
      defaultValue = "10",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      converter = NonNegativeIntegerConverter.class,
      help =
          "Flag for advanced configuration of Bazel's internal Skyframe engine. If Bazel detects"
              + " its retained heap percentage usage exceeds the threshold set by"
              + " --skyframe_high_water_mark_threshold, when a full GC event occurs, it will drop"
              + " unnecessary temporary Skyframe state, up to this many times per invocation."
              + " Defaults to 10. Zero means that full GC events will never trigger drops. If the"
              + " limit is reached, Skyframe state will no longer be dropped when a full GC event"
              + " occurs and that retained heap percentage threshold is exceeded.")
  public int skyframeHighWaterMarkFullGcDropsPerInvocation;

  @Option(
      name = "gc_thrashing_limits",
      defaultValue = "1s:2,20s:3,1m:5",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      converter = GcThrashingLimitsConverter.class,
      help =
          "Limits which, if reached, cause GcThrashingDetector to crash Bazel with an OOM. Each"
              + " limit is specified as <period>:<count> where period is a duration and count is a"
              + " positive integer. If more than --gc_thrashing_threshold percent of tenured space"
              + " (old gen heap) remains occupied after <count> consecutive full GCs within"
              + " <period>, an OOM is triggered. Multiple limits can be specified separated by"
              + " commas.")
  public ImmutableList<GcThrashingDetector.Limit> gcThrashingLimits;

  @Option(
      name = "gc_thrashing_threshold",
      oldName = "experimental_oom_more_eagerly_threshold",
      oldNameWarning = false,
      defaultValue = "100",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      converter = PercentageConverter.class,
      help =
          "The percent of tenured space occupied (0-100) above which GcThrashingDetector considers"
              + " memory pressure events against its limits (--gc_thrashing_limits). If set to 100,"
              + " GcThrashingDetector is disabled.")
  public int gcThrashingThreshold;

  @Option(
      name = "gc_churning_threshold",
      defaultValue = "100",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      converter = PercentageConverter.class,
      help =
          "At any point after an invocation has been running for at least one minute, if Blaze has"
              + " spent at least this percentage of the invocation's wall time doing full GCs,"
              + " Blaze will give up and fail with an OOM. A value of 100 effectively means to"
              + " never give up for this reason.")
  public int gcChurningThreshold;

  @Option(
      name = "gc_churning_threshold_if_multiple_top_level_targets",
      defaultValue = OptionalPercentageConverter.UNSET,
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      converter = OptionalPercentageConverter.class,
      help =
          "If set to a value in [0, 100] and this is a command that takes top-level targets (e.g."
              + " build but not query) and there are multiple such top-level targets, overrides"
              + " --gc_churning_threshold. Useful to configure more aggressive OOMing behavior"
              + " (i.e. a lower value than --gc_churning_threshold) when they are multiple"
              + " top-level targets so that the invoker of Bazel can split and retry while still"
              + " having less aggressive behavior when there is a single top-level target.")
  public OptionalInt gcChurningThresholdIfMultipleTopLevelTargets;

  // NOTE: 2024-06-11, this matches both known patterns of:
  // jdk.internal.vm.Filler[Array|Element[]] but does not match the thread
  // related jdk.internal.vm entries which we do not currently want to count.
  @Option(
      name = "jvm_heap_histogram_internal_object_pattern",
      converter = RegexPatternConverter.class,
      defaultValue = "jdk\\.internal\\.vm\\.Filler.+",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Regex for overriding the matching logic for JDK21+ JVM heap memory"
              + " collection. We are relying on volatile internal G1 GC implemenation"
              + " details to get a clean memory metric, this option allows us to adapt"
              + " to changes in that internal implementation without having to wait"
              + " for a binary release.  Passed to JDK Matcher.find()")
  public RegexPatternOption jvmHeapHistogramInternalObjectPattern;

  static final class NonNegativeIntegerConverter extends RangeConverter {
    NonNegativeIntegerConverter() {
      super(0, Integer.MAX_VALUE);
    }
  }

  static final class GcThrashingLimitsConverter
      extends Converter.Contextless<ImmutableList<GcThrashingDetector.Limit>> {
    private final CommaSeparatedOptionListConverter commaListConverter =
        new CommaSeparatedOptionListConverter();
    private final DurationConverter durationConverter = new DurationConverter();
    private final RangeConverter positiveIntConverter = new RangeConverter(1, Integer.MAX_VALUE);

    @Override
    public ImmutableList<GcThrashingDetector.Limit> convert(String input)
        throws OptionsParsingException {
      ImmutableList.Builder<GcThrashingDetector.Limit> result = ImmutableList.builder();
      for (String part : commaListConverter.convert(input)) {
        int colonIndex = part.indexOf(':');
        if (colonIndex == -1) {
          throw new OptionsParsingException("Expected <period>:<count>, got " + part);
        }
        Duration period = durationConverter.convert(part.substring(0, colonIndex));
        int count = positiveIntConverter.convert(part.substring(colonIndex + 1));
        result.add(GcThrashingDetector.Limit.of(period, count));
      }
      return result.build();
    }

    @Override
    public String getTypeDescription() {
      return "comma separated pairs of <period>:<count>";
    }
  }
}
