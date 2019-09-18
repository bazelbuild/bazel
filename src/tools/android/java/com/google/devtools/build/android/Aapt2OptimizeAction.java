// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.util.List;
import java.util.logging.Logger;

/**
 * A {@link ResourceProcessorBusyBox} action which executes {@code aapt2 optimize}, passing through
 * raw arguments.
 */
class Aapt2OptimizeAction {
  private static final Logger logger = Logger.getLogger(Aapt2OptimizeAction.class.getName());

  public static void main(String... args) throws Exception {
    logger.fine(CommandHelper.execute("Optimizing resources", buildCommand(args)));
  }

  @VisibleForTesting
  static List<String> buildCommand(String... args) {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Aapt2ConfigOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .allowResidue(true)
            .build();
    optionsParser.parseAndExitUponError(args);

    ImmutableList<String> rawOptimizeArgs =
        optionsParser.getResidue().stream()
            // TODO(b/141204955): Remove these mapping hacks once some dust has settled.
            .map(
                arg -> {
                  switch (arg) {
                    case "--shorten-resource-paths":
                    case "--enable-resource-path-shortening":
                      /* TODO(bcsf): Start using this at the time we drop AAPT2 5880457.
                      return "--shorten-resource-paths";
                      */
                      return "--enable-resource-path-shortening";
                    case "--collapse-resource-names":
                    case "--enable-resource-name-obfuscation":
                      /* TODO(bcsf): Start using this at the time we drop AAPT2 5880457.
                      return "--collapse-resource-names";
                      */
                      return "--enable-resource-obfuscation";
                    case "--resource-name-obfuscation-exemption-list":
                      // TODO(bcsf): This case goes away soon.
                      return "--whitelist-path";
                    default:
                      return arg;
                  }
                })
            .collect(toImmutableList());
    return ImmutableList.<String>builder()
        .add(optionsParser.getOptions(Aapt2ConfigOptions.class).aapt2.toString())
        .add("optimize")
        .addAll(rawOptimizeArgs)
        .build();
  }

  private Aapt2OptimizeAction() {}
}
