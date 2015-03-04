// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandUtils;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;

import java.util.Collection;
import java.util.List;

/**
 * The 'blaze canonicalize-flags' command.
 */
@Command(name = "canonicalize-flags",
         options = { CanonicalizeCommand.Options.class },
         allowResidue = true,
         mustRunInWorkspace = false,
         shortDescription = "Canonicalizes a list of %{product} options.",
         help = "This command canonicalizes a list of %{product} options. Don't forget to prepend "
             + " '--' to end option parsing before the flags to canonicalize.\n"
             + "%{options}")
public final class CanonicalizeCommand implements BlazeCommand {

  public static class CommandConverter implements Converter<String> {

    @Override
    public String convert(String input) throws OptionsParsingException {
      if (input.equals("build")) {
        return input;
      } else if (input.equals("test")) {
        return input;
      }
      throw new OptionsParsingException("Not a valid command: '" + input + "' (should be "
          + getTypeDescription() + ")");
    }

    @Override
    public String getTypeDescription() {
      return "build or test";
    }
  }

  public static class Options extends OptionsBase {

    @Option(name = "for_command",
            defaultValue = "build",
            category = "misc",
            converter = CommandConverter.class,
            help = "The command for which the options should be canonicalized.")
    public String forCommand;
  }

  @Override
  public ExitCode exec(BlazeRuntime runtime, OptionsProvider options) {
    BlazeCommand command = runtime.getCommandMap().get(
        options.getOptions(Options.class).forCommand);
    Collection<Class<? extends OptionsBase>> optionsClasses =
        BlazeCommandUtils.getOptions(
            command.getClass(), runtime.getBlazeModules(), runtime.getRuleClassProvider());
    try {
      List<String> result = OptionsParser.canonicalize(optionsClasses, options.getResidue());
      for (String piece : result) {
        runtime.getReporter().getOutErr().printOutLn(piece);
      }
    } catch (OptionsParsingException e) {
      runtime.getReporter().handle(Event.error(e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    return ExitCode.SUCCESS;
  }

  @Override
  public void editOptions(BlazeRuntime runtime, OptionsParser optionsParser) {}
}
