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
package com.google.devtools.build.lib.runtime.commands;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeCommandUtils;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.StarlarkOptionsParser;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.InvocationPolicyParser;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;

/** The 'blaze canonicalize-flags' command. */
@Command(
    name = "canonicalize-flags",
    options = {CanonicalizeCommand.Options.class, PackageCacheOptions.class},
    // inherits from query to get proper package loading options.
    inherits = {QueryCommand.class},
    allowResidue = true,
    mustRunInWorkspace = false,
    shortDescription = "Canonicalizes a list of %{product} options.",
    help =
        "This command canonicalizes a list of %{product} options. Don't forget to prepend "
            + " '--' to end option parsing before the flags to canonicalize.\n"
            + "%{options}")
public final class CanonicalizeCommand implements BlazeCommand {

  public static class Options extends OptionsBase {
    @Option(
      name = "for_command",
      defaultValue = "build",
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.TERMINAL_OUTPUT},
      help = "The command for which the options should be canonicalized."
    )
    public String forCommand;

    @Option(
      name = "invocation_policy",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.TERMINAL_OUTPUT},
      help = "Applies an invocation policy to the options to be canonicalized."
    )
    public String invocationPolicy;

    @Option(
      name = "canonicalize_policy",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Output the canonical policy, after expansion and filtering. To keep the output "
              + "clean, the canonicalized command arguments will NOT be shown when this option is "
              + "set to true. Note that the command specified by --for_command affects the "
              + "filtered policy, and if none is specified, the default command is 'build'."
    )
    public boolean canonicalizePolicy;

    @Option(
      name = "show_warnings",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.TERMINAL_OUTPUT},
      help = "Output parser warnings to standard error (e.g. for conflicting flag options)."
    )
    public boolean showWarnings;
  }

  /**
   * These options are used by the incompatible_changes_conflict_test.sh integration test, which
   * confirms that the warning for conflicting expansion options is working correctly. These flags
   * are undocumented no-ops, and are not to be used by anything outside of that test.
   */
  public static class FlagClashCanaryOptions extends OptionsBase {
    @Option(
      name = "flag_clash_canary",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP}
    )
    public boolean flagClashCanary;

    @Option(
      name = "flag_clash_canary_expander1",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      expansion = {"--flag_clash_canary=1"}
    )
    public Void flagClashCanaryExpander1;

    @Option(
      name = "flag_clash_canary_expander2",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      expansion = {"--flag_clash_canary=0"}
    )
    public Void flagClashCanaryExpander2;
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    BlazeRuntime runtime = env.getRuntime();
    Options canonicalizeOptions = options.getOptions(Options.class);
    String commandName = canonicalizeOptions.forCommand;
    BlazeCommand command = runtime.getCommandMap().get(commandName);
    if (command == null) {
      env.getReporter().handle(Event.error("Not a valid command: '" + commandName
          + "' (should be one of " + Joiner.on(", ").join(runtime.getCommandMap().keySet()) + ")"));
      return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
    }
    Collection<Class<? extends OptionsBase>> optionsClasses =
        ImmutableList.<Class<? extends OptionsBase>>builder()
            .addAll(BlazeCommandUtils.getOptions(
                command.getClass(), runtime.getBlazeModules(), runtime.getRuleClassProvider()))
            .add(FlagClashCanaryOptions.class)
            .build();

    // set up the package cache for starlark options parsing
    try {
      env.setupPackageCache(options);
    } catch (InterruptedException e) {
      env.getReporter().handle(Event.error("canonicalization interrupted"));
      return BlazeCommandResult.exitCode(ExitCode.INTERRUPTED);
    } catch (AbruptExitException e) {
      env.getReporter().handle(Event.error(null, "Unknown error: " + e.getMessage()));
      return BlazeCommandResult.exitCode(e.getExitCode());
    }

    try {
      OptionsParser parser =
          OptionsParser.builder()
              .optionsClasses(optionsClasses)
              .skipStarlarkOptionPrefixes()
              .allowResidue(true)
              .build();
      parser.parse(options.getResidue());
      StarlarkOptionsParser.newStarlarkOptionsParser(env, parser).parse(env.getReporter());
      if (!parser.getResidue().isEmpty()) {
        String errorMsg = "Unrecognized arguments: " + Joiner.on(' ').join(parser.getResidue());
        throw new OptionsParsingException(errorMsg);
      }

      InvocationPolicy policy =
          InvocationPolicyParser.parsePolicy(canonicalizeOptions.invocationPolicy);
      InvocationPolicyEnforcer invocationPolicyEnforcer =
          new InvocationPolicyEnforcer(policy, Level.INFO);
      invocationPolicyEnforcer.enforce(parser, commandName);

      if (canonicalizeOptions.showWarnings) {
        for (String warning : parser.getWarnings()) {
          env.getReporter().handle(Event.warn(warning));
        }
      }

      // Print out the canonical invocation policy if requested.
      if (canonicalizeOptions.canonicalizePolicy) {
        InvocationPolicy effectivePolicy =
            InvocationPolicyEnforcer.getEffectiveInvocationPolicy(
                policy, parser, commandName, Level.INFO);
        env.getReporter().getOutErr().printOutLn(effectivePolicy.toString());
      } else {
        // Otherwise, print out the canonical command line
        List<String> nativeResult = parser.canonicalize();
        ImmutableList.Builder<String> result = ImmutableList.<String>builder().addAll(nativeResult);
        for (Map.Entry<String, Object> starlarkOption : parser.getStarlarkOptions().entrySet()) {
          result.add("--" + starlarkOption.getKey() + "=" + starlarkOption.getValue());
        }
        for (String piece : result.build()) {
          env.getReporter().getOutErr().printOutLn(piece);
        }
      }
    } catch (OptionsParsingException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
    }
    return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
  }
}
