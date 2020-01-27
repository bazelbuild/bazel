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
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
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
    this.env = null;
    this.autoOutputFilter = null;
  }

  @Subscribe
  @SuppressWarnings("unused")
  public void buildStarting(BuildStartingEvent event) {
    BuildRequestOptions requestOptions = env.getOptions().getOptions(BuildRequestOptions.class);
    Pattern outputFilter =
        (requestOptions != null) && (requestOptions.outputFilter != null)
            ? requestOptions.outputFilter.regexPattern()
            : null;
    if (outputFilter != null) {
      // Coarse-grained initialization of the output filter. This only has an
      // effect if the --output_filter option is given. The auto output filter is
      // only initialized later, when we know all targets. For now this is good
      // enough, as the target parsing only loads packages that are mentioned on
      // the command line (which are included by all auto output filters).
      env.getReporter().setOutputFilter(RegexOutputFilter.forPattern(outputFilter));
    } else {
      this.autoOutputFilter = env.getOptions().getOptions(Options.class).autoOutputFilter;
    }
  }

  @Subscribe
  @SuppressWarnings("unused")
  public void targetParsingComplete(TargetParsingCompleteEvent event) {
    if (autoOutputFilter != null) {
      env.getReporter().setOutputFilter(autoOutputFilter.getFilter(event.getLabels()));
    }
  }
}
