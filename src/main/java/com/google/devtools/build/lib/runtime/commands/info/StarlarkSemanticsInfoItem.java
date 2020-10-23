// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands.info;

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.common.options.OptionsParsingResult;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Info item for the effective current set of Starlark semantics option values.
 *
 * <p>This is hidden because its output is verbose and may be multiline.
 */
public final class StarlarkSemanticsInfoItem extends InfoItem {
  private final OptionsParsingResult commandOptions;

  public StarlarkSemanticsInfoItem(OptionsParsingResult commandOptions) {
    super(
        /*name=*/ "starlark-semantics",
        /*description=*/ "The effective set of Starlark semantics option values.",
        /*hidden=*/ true);
    this.commandOptions = commandOptions;
  }

  @Override
  public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
    BuildLanguageOptions buildLanguageOptions =
        commandOptions.getOptions(BuildLanguageOptions.class);
    SkyframeExecutor skyframeExecutor = env.getBlazeWorkspace().getSkyframeExecutor();
    StarlarkSemantics effectiveStarlarkSemantics =
        skyframeExecutor.getEffectiveStarlarkSemantics(buildLanguageOptions);
    return print(effectiveStarlarkSemantics);
  }
}
