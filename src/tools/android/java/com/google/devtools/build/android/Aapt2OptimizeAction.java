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

  private static List<String> buildCommand(String... args) {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Aapt2ConfigOptions.class, ResourceProcessorCommonOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .allowResidue(true)
            .build();
    optionsParser.parseAndExitUponError(args);

    return ImmutableList.<String>builder()
        .add(optionsParser.getOptions(Aapt2ConfigOptions.class).aapt2.toString())
        .add("optimize")
        .addAll(optionsParser.getResidue())
        .build();
  }

  private Aapt2OptimizeAction() {}
}
