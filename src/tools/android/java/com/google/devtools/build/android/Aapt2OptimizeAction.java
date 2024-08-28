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

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
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

  @Parameters(separators = "= ")
  static class Options extends OptionsBaseWithResidue {
    // NOTE: This options class needs the ParametersDelegate feature from JCommander for several
    // reasons:
    // * There are no Aapt2OptimizeAction-specific options, just fake "inheritance" from
    //   Aapt2ConfigOptions and ResourceProcessorCommonOptions.
    // * The action itself reads the residue from the command line
    // In JCommander, residue collection is done at a per-option-class basis, not per OptionsParser,
    // as was the case with the Bazel OptionsParser. Simultaneously, only _one_ object in a list of
    // Objects passed to JCommander can have residue collection.
    // Therefore, the most straightforward option here is to "inherit" the sub-option classes into a
    // super Options class, with residue collection enabled for Options.

    @ParametersDelegate public Aapt2ConfigOptions aapt2ConfigOptions = new Aapt2ConfigOptions();

    @ParametersDelegate
    public ResourceProcessorCommonOptions resourceProcessorCommonOptions =
        new ResourceProcessorCommonOptions();
  }

  private static List<String> buildCommand(String... args) throws CompatOptionsParsingException {
    Options options = new Options();
    JCommander jc = new JCommander(options);
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(options, preprocessedArgs);
    jc.parse(normalizedArgs);

    return ImmutableList.<String>builder()
        .add(options.aapt2ConfigOptions.aapt2.toString())
        .add("optimize")
        .addAll(options.getResidue())
        .build();
  }

  private Aapt2OptimizeAction() {}
}
