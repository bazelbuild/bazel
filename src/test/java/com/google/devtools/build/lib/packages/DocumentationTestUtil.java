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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertWithMessage;

import com.google.devtools.build.docgen.DocCheckerUtils;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.exec.TestPolicy;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.runtime.commands.BuiltinCommandModule;
import com.google.devtools.build.lib.runtime.commands.RunCommand;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Utility functions for validating correctness of Bazel documentation. */
abstract class DocumentationTestUtil {

  private static final class DummyBuiltinCommandModule extends BuiltinCommandModule {
    DummyBuiltinCommandModule() {
      super(new RunCommand(TestPolicy.EMPTY_POLICY));
    }
  }

  private DocumentationTestUtil() {}

  private static final Pattern CODE_FLAG_PATTERN =
      Pattern.compile(
          "<code class\\s*=\\s*[\"']flag[\"']\\s*>--([a-z_\\[\\]]*)<\\/code>",
          Pattern.CASE_INSENSITIVE);

  /**
   * Validates that a user manual {@code documentationSource} contains only the flags actually
   * provided by a given set of modules.
   */
  static void validateUserManual(
      List<Class<? extends BlazeModule>> modules,
      ConfiguredRuleClassProvider ruleClassProvider,
      String documentationSource,
      Set<String> extraValidOptions)
      throws Exception {
    // if there is a class missing, one can find it using
    //   find . -name "*.java" -exec grep -Hn "@Option(name = " {} \; | grep "xxx"
    // where 'xxx' is a flag name.
    List<BlazeModule> blazeModules = BlazeRuntime.createModules(modules);

    Set<String> validOptions = new HashSet<>();

    var startupOptionsClasses = BlazeCommandUtils.getStartupOptions(blazeModules);
    // collect all startup options
    for (Class<? extends OptionsBase> optionsClass : startupOptionsClasses) {
      validOptions.addAll(Options.getDefaults(optionsClass).asMap().keySet());
    }
    validOptions.addAll(extraValidOptions);

    var startupOptions = OptionsParser.builder().optionsClasses(startupOptionsClasses).build();
    startupOptions.parse();

    // collect all command options
    ServerBuilder serverBuilder = new ServerBuilder();
    new DummyBuiltinCommandModule().serverInit(startupOptions, serverBuilder);
    for (BlazeModule module : blazeModules) {
      module.serverInit(startupOptions, serverBuilder);
    }
    List<BlazeCommand> blazeCommands = serverBuilder.getCommands();

    for (BlazeCommand command : blazeCommands) {
      for (Class<? extends OptionsBase> optionClass :
          BlazeCommandUtils.getOptions(command.getClass(), blazeModules, ruleClassProvider)) {
        validOptions.addAll(Options.getDefaults(optionClass).asMap().keySet());
      }
    }

    // check validity of option flags in manual
    Matcher anchorMatcher = CODE_FLAG_PATTERN.matcher(documentationSource);
    String flag;
    boolean found;

    while (anchorMatcher.find()) {
      flag = anchorMatcher.group(1);
      found = validOptions.contains(flag);
      if (!found && flag.startsWith("no")) {
        found = validOptions.contains(flag.substring(2));
      }
      if (!found && flag.startsWith("[no]")) {
        found = validOptions.contains(flag.substring(4));
      }

      assertWithMessage("flag '" + flag + "' is not a bazel option (anymore)").that(found).isTrue();
    }

    String unclosedTag = DocCheckerUtils.getFirstUnclosedTagAndPrintHelp(documentationSource);
    assertWithMessage("Unclosed tag found: " + unclosedTag).that(unclosedTag).isNull();
  }
}
