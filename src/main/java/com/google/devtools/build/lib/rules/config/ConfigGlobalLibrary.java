// Copyright 2018 The Bazel Authors. All rights reserved.
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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import static com.google.devtools.build.lib.cmdline.LabelConstants.COMMAND_LINE_OPTION_PREFIX;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition.Settings;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigGlobalLibraryApi;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigurationTransitionApi;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Implementation of {@link ConfigGlobalLibraryApi}.
 *
 * <p>A collection of top-level Starlark functions pertaining to configuration.
 */
public class ConfigGlobalLibrary implements ConfigGlobalLibraryApi {

  @Override
  public ConfigurationTransitionApi transition(
      StarlarkCallable implementation,
      Sequence<?> inputs, // <String> expected
      Sequence<?> outputs, // <String> expected
      StarlarkThread thread)
      throws EvalException {
    StarlarkSemantics semantics = thread.getSemantics();
    BazelModuleContext moduleContext = BazelModuleContext.ofInnermostBzlOrThrow(thread);

    List<String> inputsList = Sequence.cast(inputs, String.class, "inputs");
    List<String> outputsList = Sequence.cast(outputs, String.class, "outputs");
    validateBuildSettingKeys(
        inputsList,
        Settings.INPUTS,
        /* allowIncompatibleAndExperimentalOptions= */ false,
        moduleContext.packageContext());
    validateBuildSettingKeys(
        outputsList,
        Settings.OUTPUTS,
        /* allowIncompatibleAndExperimentalOptions= */ false,
        moduleContext.packageContext());
    Location location = thread.getCallerLocation();
    return StarlarkDefinedConfigTransition.newRegularTransition(
        implementation,
        inputsList,
        outputsList,
        semantics,
        moduleContext.label(),
        location,
        moduleContext.repoMapping());
  }

  @Override
  public ConfigurationTransitionApi execTransition(
      StarlarkCallable implementation,
      Sequence<?> inputs, // <String> expected
      Sequence<?> outputs, // <String> expected
      StarlarkThread thread)
      throws EvalException {
    // TODO: blaze-configurability-team - When we relax this, add checks that regular usages of
    // transitions (ie, rule and attribute) cannot use exec_transition.
    BuiltinRestriction.failIfCalledOutsideBuiltins(thread);

    StarlarkSemantics semantics = thread.getSemantics();
    BazelModuleContext moduleContext = BazelModuleContext.ofInnermostBzlOrThrow(thread);

    List<String> inputsList = Sequence.cast(inputs, String.class, "inputs");
    List<String> outputsList = Sequence.cast(outputs, String.class, "outputs");
    validateBuildSettingKeys(
        inputsList,
        Settings.INPUTS,
        /* allowIncompatibleAndExperimentalOptions= */ true,
        moduleContext.packageContext());
    validateBuildSettingKeys(
        outputsList,
        Settings.OUTPUTS,
        /* allowIncompatibleAndExperimentalOptions= */ true,
        moduleContext.packageContext());
    Location location = thread.getCallerLocation();
    return StarlarkDefinedConfigTransition.newExecTransition(
        implementation,
        inputsList,
        outputsList,
        semantics,
        moduleContext.label(),
        location,
        moduleContext.repoMapping());
  }

  // TODO(b/237422931): move into testing module
  @Override
  public ConfigurationTransitionApi analysisTestTransition(
      Dict<?, ?> changedSettings, // <String, String> expected
      StarlarkThread thread)
      throws EvalException {
    BazelModuleContext moduleContext = BazelModuleContext.ofInnermostBzlOrThrow(thread);
    Map<String, Object> changedSettingsMap =
        Dict.cast(changedSettings, String.class, Object.class, "changed_settings dict");
    validateBuildSettingKeys(
        changedSettingsMap.keySet(),
        Settings.OUTPUTS,
        /* allowIncompatibleAndExperimentalOptions= */ true,
        moduleContext.packageContext());
    Location location = thread.getCallerLocation();
    return StarlarkDefinedConfigTransition.newAnalysisTestTransition(
        changedSettingsMap, moduleContext.repoMapping(), moduleContext.label(), location);
  }

  private void validateBuildSettingKeys(
      Iterable<String> optionKeys,
      Settings keyErrorDescriptor,
      boolean allowIncompatibleAndExperimentalOptions,
      Label.PackageContext packageContext)
      throws EvalException {

    HashSet<String> processedOptions = Sets.newHashSet();
    String singularErrorDescriptor = keyErrorDescriptor == Settings.INPUTS ? "input" : "output";

    for (String optionKey : optionKeys) {
      if (!optionKey.startsWith(COMMAND_LINE_OPTION_PREFIX)) {
        try {
          Label label = Label.parseWithRepoContext(optionKey, packageContext);
          if (!label.getRepository().isVisible()) {
            throw Starlark.errorf(
                "invalid transition %s '%s': no repo visible as @%s from %s",
                singularErrorDescriptor,
                label,
                label.getRepository().getName(),
                label.getRepository().getContextRepoDisplayString());
          }
        } catch (LabelSyntaxException e) {
          throw Starlark.errorf(
              "invalid transition %s '%s'. If this is intended as a native option, "
                  + "it must begin with //command_line_option: %s",
              singularErrorDescriptor, optionKey, e.getMessage());
        }
      } else {
        String optionName = optionKey.substring(COMMAND_LINE_OPTION_PREFIX.length());
        if (!allowIncompatibleAndExperimentalOptions && !validOptionName(optionName)) {
          throw Starlark.errorf(
              "Invalid transition %s '%s'. Cannot transition on --experimental_* or "
                  + "--incompatible_* options",
              singularErrorDescriptor, optionKey);
        }
      }
      if (!processedOptions.add(optionKey)) {
        throw Starlark.errorf("duplicate transition %s '%s'", singularErrorDescriptor, optionKey);
      }
    }
  }

  /**
   * Flags that user-defined transitions aren't allowed to set.
   *
   * <p>Exec transitions are exempt from this because they already set many non-standard flags.
   * Maybe that can change in a future migration, but that's their current semantics. See caller
   * code for implementation details.
   */
  private static boolean validOptionName(String optionName) {
    if (optionName.startsWith("experimental_")) {
      // Don't allow experimental flags.
      return false;
    }

    if (optionName.equals("incompatible_enable_cc_toolchain_resolution")
        || optionName.equals("incompatible_enable_apple_toolchain_resolution")) {
      // This is specifically allowed.
      return true;
    } else if (optionName.startsWith("incompatible_")) {
      // Don't allow other incompatible flags.
      return false;
    }

    return true;
  }
}
