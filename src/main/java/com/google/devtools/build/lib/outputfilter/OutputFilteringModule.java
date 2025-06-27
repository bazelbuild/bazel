// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.outputfilter;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.events.OutputFilter.RegexOutputFilter;
import com.google.devtools.build.lib.pkgcache.TargetParsingCompleteEvent;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Blaze module implementing output filtering.
 */
public final class OutputFilteringModule extends BlazeModule {
  /**
   * Options controlling output filtering.
   */
  public static class Options extends OptionsBase {
    @Option(
        name = "auto_output_filter",
        converter = AutoOutputFilter.Converter.class,
        defaultValue = "none",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "If --output_filter is not specified, then the value for this option is used "
                + "create a filter automatically. Allowed values are 'none' (filter nothing "
                + "/ show everything), 'all' (filter everything / show nothing), 'packages' "
                + "(include output from rules in packages mentioned on the Blaze command line), "
                + "and 'subpackages' (like 'packages', but also include subpackages). For the "
                + "'packages' and 'subpackages' values //java/foo and //javatests/foo are treated "
                + "as one package)'.")
    public AutoOutputFilter autoOutputFilter;

    @Option(
        name = "output_suppression",
        allowMultiple = true,
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.LOGGING,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Specifies a suppression rule for build output.")
    public List<String> outputSuppressions;
  }

  private CommandEnvironment env;
  private AutoOutputFilter autoOutputFilter;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name()) ? ImmutableList.of(Options.class) : ImmutableList.of();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    if (env != null) {
      // TODO: remove debug logging
      // env.getReporter().handle(Event.info("afterCommand() called"));
      // env.getReporter()
      //     .handle(
      //         Event.info(
      //             "Final output filter is "
      //                 + env.getReporter().getOutputFilter().getClass().getName()));
      if (env.getReporter().getOutputFilter() instanceof OutputSuppressionFilter) {
        OutputSuppressionFilter filter =
            (OutputSuppressionFilter) env.getReporter().getOutputFilter();
        filter.verifyCounts(env.getReporter());
        int totalSuppressed = filter.getTotalSuppressedCount();
        env.getReporter()
            .handle(
                Event.info(
                    String.format(
                        "Suppressed %d messages via --output_suppression; pass"
                            + " --no_output_suppression to display.",
                        totalSuppressed)));
      }
    }
    this.env = null;
    this.autoOutputFilter = null;
  }

  @Subscribe
  @SuppressWarnings("unused")
  public void buildStarting(BuildStartingEvent event) {
    Options options = env.getOptions().getOptions(Options.class);
    BuildRequestOptions requestOptions = env.getOptions().getOptions(BuildRequestOptions.class);

    boolean suppressionsActive =
        options.outputSuppressions != null && !options.outputSuppressions.isEmpty();
    boolean outputFilterActive =
        requestOptions != null && requestOptions.outputFilter != null;
    boolean autoFilterActive =
        options.autoOutputFilter != null && options.autoOutputFilter != AutoOutputFilter.NONE;

    if (suppressionsActive) {
      // TODO: remove debug logging
      // env.getReporter().handle(Event.info("Installing OutputSuppressionFilter"));
      // env.getReporter()
      //     .handle(
      //         Event.info(
      //             String.format(
      //                 "Activating %d output suppression rules.", options.outputSuppressions.size())));
      env.getReporter()
          .setOutputFilter(new OutputSuppressionFilter(options.outputSuppressions));
      this.autoOutputFilter = null;

      if (outputFilterActive) {
        env.getReporter()
            .handle(
                Event.warn(
                    "Both --output_suppression and --output_filter are specified. "
                        + "--output_suppression takes precedence."));
      }
      if (autoFilterActive) {
        env.getReporter()
            .handle(
                Event.warn(
                    "Both --output_suppression and --auto_output_filter are specified. "
                        + "--output_suppression takes precedence."));
      }
      return;
    }

    if (outputFilterActive) {
      // TODO: remove debug logging
      // env.getReporter().handle(Event.info("Installing RegexOutputFilter"));
      env.getReporter()
          .setOutputFilter(RegexOutputFilter.forPattern(requestOptions.outputFilter.regexPattern()));
    } else {
      // TODO: remove debug logging
      // env.getReporter().handle(Event.info("Using auto output filter"));
      this.autoOutputFilter = env.getOptions().getOptions(Options.class).autoOutputFilter;
    }

  }

  @Subscribe
  @SuppressWarnings("unused")
  public void targetParsingComplete(TargetParsingCompleteEvent event) {
    if (autoOutputFilter != null) {
      // TODO: remove debug logging
      // env.getReporter()
      //     .handle(
      //         Event.info(
      //             "Installing auto output filter "
      //                 + autoOutputFilter.getClass().getName()
      //                 + " in targetParsingComplete"));
      env.getReporter().setOutputFilter(autoOutputFilter.getFilter(event.getLabels()));
    }
  }
}
