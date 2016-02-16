// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandUtils;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implementation of the dump command.
 */
@Command(allowResidue = false,
         mustRunInWorkspace = false,
         options = { DumpCommand.DumpOptions.class },
         help = "Usage: %{product} dump <options>\n"
         + "Dumps the internal state of the %{product} server process.  This command is provided "
         + "as an aid to debugging, not as a stable interface, so users should not try to "
         + "parse the output; instead, use 'query' or 'info' for this purpose.\n%{options}",
         name = "dump",
         shortDescription = "Dumps the internal state of the %{product} server process.")
public class DumpCommand implements BlazeCommand {

  /**
   * NB! Any changes to this class must be kept in sync with anyOutput variable
   * value in the {@link DumpCommand#exec(CommandEnvironment,OptionsProvider)} method below.
   */
  public static class DumpOptions extends OptionsBase {

    @Option(name = "packages",
        defaultValue = "false",
        category = "verbosity",
        help = "Dump package cache content.")
    public boolean dumpPackages;

    @Option(name = "vfs",
        defaultValue = "false",
        category = "verbosity",
        help = "Dump virtual filesystem cache content.")
    public boolean dumpVfs;

    @Option(name = "action_cache",
        defaultValue = "false",
        category = "verbosity",
        help = "Dump action cache content.")
    public boolean dumpActionCache;

    @Option(name = "rule_classes",
        defaultValue = "false",
        category = "verbosity",
        help = "Dump rule classes.")
    public boolean dumpRuleClasses;

    @Option(name = "skyframe",
        defaultValue = "off",
        category = "verbosity",
        converter = SkyframeDumpEnumConverter.class,
        help = "Dump Skyframe graph: 'off', 'summary', or 'detailed'.")
    public SkyframeDumpOption dumpSkyframe;
  }

  /**
   * Different ways to dump information about Skyframe.
   */
  public enum SkyframeDumpOption {
    OFF,
    SUMMARY,
    DETAILED;
  }

  /**
   * Enum converter for SkyframeDumpOption.
   */
  public static class SkyframeDumpEnumConverter extends EnumConverter<SkyframeDumpOption> {
    public SkyframeDumpEnumConverter() {
      super(SkyframeDumpOption.class, "Skyframe Dump option");
    }
  }

  @Override
  public void editOptions(CommandEnvironment env, OptionsParser optionsParser) {}

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    BlazeRuntime runtime = env.getRuntime();
    DumpOptions dumpOptions = options.getOptions(DumpOptions.class);

    boolean anyOutput =
        dumpOptions.dumpPackages
            || dumpOptions.dumpVfs
            || dumpOptions.dumpActionCache
            || dumpOptions.dumpRuleClasses
            || (dumpOptions.dumpSkyframe != SkyframeDumpOption.OFF);
    if (!anyOutput) {
      Map<String, String> categories = new HashMap<>();
      categories.put("verbosity", "Options that control what internal state is dumped");
      Collection<Class<? extends OptionsBase>> optionList = new ArrayList<>();
      optionList.add(DumpOptions.class);

      env.getReporter().getOutErr().printErrLn(BlazeCommandUtils.expandHelpTopic(
          getClass().getAnnotation(Command.class).name(),
          getClass().getAnnotation(Command.class).help(),
          getClass(),
          optionList, categories, OptionsParser.HelpVerbosity.LONG));
      return ExitCode.ANALYSIS_FAILURE;
    }
    PrintStream out = new PrintStream(env.getReporter().getOutErr().getOutputStream());
    try {
      out.println("Warning: this information is intended for consumption by developers");
      out.println("only, and may change at any time.  Script against it at your own risk!");
      out.println();
      boolean success = true;

      if (dumpOptions.dumpPackages) {
        env.getPackageManager().dump(out);
        out.println();
      }

      if (dumpOptions.dumpVfs) {
        out.println("Filesystem cache");
        FileSystemUtils.dump(runtime.getOutputBase().getFileSystem(), out);
        out.println();
      }

      if (dumpOptions.dumpActionCache) {
        success &= dumpActionCache(env, out);
        out.println();
      }

      if (dumpOptions.dumpRuleClasses) {
        dumpRuleClasses(runtime, out);
        out.println();
      }

      if (dumpOptions.dumpSkyframe != SkyframeDumpOption.OFF) {
        success &= dumpSkyframe(runtime, dumpOptions.dumpSkyframe == SkyframeDumpOption.SUMMARY,
            out);
        out.println();
      }

      return success ? ExitCode.SUCCESS : ExitCode.ANALYSIS_FAILURE;

    } finally {
      out.flush();
    }
  }

  private boolean dumpActionCache(CommandEnvironment env, PrintStream out) {
    try {
      env.getPersistentActionCache().dump(out);
    } catch (IOException e) {
      env.getReporter().handle(Event.error("Cannot dump action cache: " + e.getMessage()));
      return false;
    }
    return true;
  }

  private boolean dumpSkyframe(BlazeRuntime runtime, boolean summarize, PrintStream out) {
    runtime.getSkyframeExecutor().dump(summarize, out);
    return true;
  }

  private void dumpRuleClasses(BlazeRuntime runtime, PrintStream out) {
    PackageFactory factory = runtime.getPackageFactory();
    List<String> ruleClassNames = new ArrayList<>(factory.getRuleClassNames());
    Collections.sort(ruleClassNames);
    for (String name : ruleClassNames) {
      if (name.startsWith("$")) {
        continue;
      }
      RuleClass ruleClass = factory.getRuleClass(name);
      out.print(ruleClass + "(");
      boolean first = true;
      for (Attribute attribute : ruleClass.getAttributes()) {
        if (attribute.isImplicit()) {
          continue;
        }
        if (first) {
          first = false;
        } else {
          out.print(", ");
        }
        out.print(attribute.getName());
      }
      out.println(")");
    }
  }
}
