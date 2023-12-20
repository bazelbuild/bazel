// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime.commands;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.skyframe.InMemoryGraphImpl;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeFocuser;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.PrintStream;

/** The focus command. */
@Command(
    hidden = true, // experimental, don't show in the help command.
    options = {
      FocusCommand.FocusOptions.class,
    },
    help =
        "Usage: %{product} focus <options>\n"
            + "Reduces the memory usage of the %{product} JVM by promising that the user will only "
            + "change a given set of files."
            + "\n%{options}",
    name = "focus",
    shortDescription = "EXPERIMENTAL. Reduce memory usage with working sets.")
public class FocusCommand implements BlazeCommand {

  /** The set of options for the focus command. */
  public static class FocusOptions extends OptionsBase {
    @Option(
        name = "dump_keys",
        defaultValue = "false",
        effectTags = OptionEffectTag.TERMINAL_OUTPUT,
        // Deliberately undocumented.
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        help = "Dump the focused SkyKeys.")
    public boolean dumpKeys;
  }

  /**
   * Reports the computed set of SkyKeys that need to be kept in the Skyframe graph for incremental
   * correctness.
   *
   * @param reporter the event reporter
   * @param focusResult the result from SkyframeFocuser
   */
  private void dumpKeys(Reporter reporter, SkyframeFocuser.FocusResult focusResult) {
    try (PrintStream pos = new PrintStream(reporter.getOutErr().getOutputStream())) {
      pos.printf("Rdeps kept:\n");
      for (SkyKey key : focusResult.getRdeps()) {
        pos.printf("  %s", key.getCanonicalName());
      }
      pos.printf("\n");
      pos.printf("Deps kept:\n");
      for (SkyKey key : focusResult.getDeps()) {
        pos.printf("  %s", key.getCanonicalName());
      }
    }
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    env.getReporter()
        .handle(
            Event.warn(
                "The focus command is experimental. Feel free to test it, "
                    + "but do not depend on it yet."));
    env.getReporter().handle(Event.info("Focusing..."));

    FocusOptions focusOptions = options.getOptions(FocusOptions.class);

    SkyframeExecutor executor = env.getSkyframeExecutor();
    InMemoryMemoizingEvaluator evaluator = (InMemoryMemoizingEvaluator) executor.getEvaluator();
    InMemoryGraphImpl graph = (InMemoryGraphImpl) evaluator.getInMemoryGraph();
    SkyframeFocuser.FocusResult focusResult = SkyframeFocuser.focus(graph, env.getReporter());

    if (focusOptions.dumpKeys) {
      dumpKeys(env.getReporter(), focusResult);
    }

    // Always succeeds (for now).
    return BlazeCommandResult.success();
  }
}
