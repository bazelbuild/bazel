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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition.COMMAND_LINE_OPTION_PREFIX;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.VerifyException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationValueEvent;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.OptionInfo;
import com.google.devtools.build.lib.analysis.config.transitions.BaselineOptionsValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/** A builder for {@link BuildConfigurationValue} instances. */
public final class BuildConfigurationFunction implements SkyFunction {

  // The length of the hash of the config tacked onto the end of the output path.
  // Limited for ergonomics and MAX_PATH reasons.
  private static final int HASH_LENGTH = 12;

  private final BlazeDirectories directories;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final FragmentFactory fragmentFactory = new FragmentFactory();

  public BuildConfigurationFunction(
      BlazeDirectories directories, RuleClassProvider ruleClassProvider) {
    this.directories = directories;
    this.ruleClassProvider = (ConfiguredRuleClassProvider) ruleClassProvider;
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, BuildConfigurationFunctionException {
    WorkspaceNameValue workspaceNameValue = (WorkspaceNameValue) env
        .getValue(WorkspaceNameValue.key());
    if (workspaceNameValue == null) {
      return null;
    }

    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }
    BuildConfigurationKey key = (BuildConfigurationKey) skyKey.argument();

    BuildOptions targetOptions = key.getOptions();
    CoreOptions coreOptions = targetOptions.get(CoreOptions.class);

    String transitionDirectoryNameFragment;
    if (targetOptions.hasNoConfig()) {
      transitionDirectoryNameFragment = "noconfig"; // See NoConfigTransition.
    } else if (coreOptions.useBaselineForOutputDirectoryNamingScheme()) {
      boolean applyExecTransitionToBaseline =
          coreOptions.outputDirectoryNamingScheme.equals(
                  CoreOptions.OutputDirectoryNamingScheme.DIFF_AGAINST_DYNAMIC_BASELINE)
              && coreOptions.isExec;
      var baselineOptionsValue =
          (BaselineOptionsValue)
              env.getValue(BaselineOptionsValue.key(applyExecTransitionToBaseline));
      if (baselineOptionsValue == null) {
        return null;
      }

      transitionDirectoryNameFragment =
          computeNameFragmentWithDiff(targetOptions, baselineOptionsValue.toOptions());
    } else {
      transitionDirectoryNameFragment =
          computeNameFragmentWithAffectedByStarlarkTransition(targetOptions);
    }

    try {
      var configurationValue =
          BuildConfigurationValue.create(
              targetOptions,
              RepositoryName.createUnvalidated(workspaceNameValue.getName()),
              starlarkSemantics.getBool(
                  BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT),
              transitionDirectoryNameFragment,
              // Arguments below this are server-global.
              directories,
              ruleClassProvider,
              fragmentFactory);
      env.getListener().post(ConfigurationValueEvent.create(configurationValue));
      return configurationValue;
    } catch (InvalidConfigurationException e) {
      throw new BuildConfigurationFunctionException(e);
    }
  }

  private static final class BuildConfigurationFunctionException extends SkyFunctionException {
    BuildConfigurationFunctionException(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }

  /**
   * Compute the hash for the new BuildOptions based on the names and values of all options (both
   * native and Starlark) that are different from some supplied baseline configuration.
   */
  @VisibleForTesting
  public static String computeNameFragmentWithDiff(
      BuildOptions toOptions, BuildOptions baselineOptions) {
    // Quick short-circuit for trivial case.
    if (toOptions.equals(baselineOptions)) {
      return "";
    }

    // TODO(blaze-configurability-team): As a mild performance update, getFirst already includes
    //   details of the corresponding option. Could incorporate this instead of hashChosenOptions
    //   regenerating the OptionDefinitions and values.
    BuildOptions.OptionsDiff diff = BuildOptions.diff(toOptions, baselineOptions);
    // Note: getFirst only excludes options trimmed between baselineOptions to toOptions and this is
    //   considered OK as a given Rule should not be being built with options of different
    //   trimmings. See longform note in {@link ConfiguredTargetKey} for details.
    ImmutableSet<String> chosenNativeOptions =
        diff.getFirst().keySet().stream()
            .filter(
                optionDef ->
                    !optionDef.hasOptionMetadataTag(OptionMetadataTag.EXPLICIT_IN_OUTPUT_PATH))
            .map(OptionDefinition::getOptionName)
            .collect(toImmutableSet());
    // Note: getChangedStarlarkOptions includes all changed options, added options and removed
    //   options between baselineOptions and toOptions. This is necessary since there is no current
    //   notion of trimming a Starlark option: 'null' or non-existent justs means set to default.
    ImmutableSet<String> chosenStarlarkOptions =
        diff.getChangedStarlarkOptions().stream().map(Label::toString).collect(toImmutableSet());
    return hashChosenOptions(toOptions, chosenNativeOptions, chosenStarlarkOptions);
  }

  /**
   * Compute the output directory name fragment corresponding to the new BuildOptions based on the
   * names and values of all options (both native and Starlark) previously transitioned anywhere in
   * the build by Starlark transitions. Options only set on command line are not affecting the
   * computation.
   *
   * @param toOptions the {@link BuildOptions} to use to calculate which we need to compute {@code
   *     transitionDirectoryNameFragment}.
   */
  private static String computeNameFragmentWithAffectedByStarlarkTransition(
      BuildOptions toOptions) {
    CoreOptions buildConfigOptions = toOptions.get(CoreOptions.class);
    if (buildConfigOptions.affectedByStarlarkTransition.isEmpty()) {
      return "";
    }

    ImmutableList.Builder<String> affectedNativeOptions = ImmutableList.builder();
    ImmutableList.Builder<String> affectedStarlarkOptions = ImmutableList.builder();

    for (String optionName : buildConfigOptions.affectedByStarlarkTransition) {
      if (optionName.startsWith(COMMAND_LINE_OPTION_PREFIX)) {
        String nativeOptionName = optionName.substring(COMMAND_LINE_OPTION_PREFIX.length());
        affectedNativeOptions.add(nativeOptionName);
      } else {
        affectedStarlarkOptions.add(optionName);
      }
    }

    return hashChosenOptions(
        toOptions, affectedNativeOptions.build(), affectedStarlarkOptions.build());
  }

  /**
   * Compute a hash of the given BuildOptions by hashing only the options referenced in both
   * chosenNative and chosenStarlark. The order of the chosen order does not matter (as this
   * function will effectively sort them into a canonical order) and the pre-hash for each option
   * will be of the form (//command_line_option:[native option]|[Starlark option label])=[value].
   *
   * <p>If a supplied native option does not exist, it is skipped (as it is presumed non-existence
   * is due to trimming).
   *
   * <p>If a supplied Starlark option does exist, the pre-hash will be [Starlark option label]@null
   * (as it is presumed non-existence is due to being set to default value).
   */
  private static String hashChosenOptions(
      BuildOptions toOptions, Iterable<String> chosenNative, Iterable<String> chosenStarlark) {
    // TODO(blaze-configurability-team): A mild performance optimization would have this be global.
    ImmutableMap<String, OptionInfo> optionInfoMap = OptionInfo.buildMapFrom(toOptions);

    // Note that the TreeMap guarantees a stable ordering of keys and thus
    // it is okay if chosenNative or chosenStarlark do not have a stable iteration order
    TreeMap<String, Object> toHash = new TreeMap<>();
    for (String nativeOptionName : chosenNative) {
      Object value;
      try {
        OptionInfo optionInfo = optionInfoMap.get(nativeOptionName);
        if (optionInfo == null) {
          // This can occur if toOptions has been trimmed but the supplied chosen native options
          // includes that trimmed options.
          // (e.g. legacy naming mode, using --trim_test_configuration and --test_arg transition).
          continue;
        }
        value =
            optionInfo
                .getDefinition()
                .getField()
                .get(toOptions.get(optionInfoMap.get(nativeOptionName).getOptionClass()));
      } catch (IllegalAccessException e) {
        throw new VerifyException(
            "IllegalAccess for option " + nativeOptionName + ": " + e.getMessage());
      }
      // TODO(blaze-configurability-team): The commandline option is legacy and can be removed
      //   after fixing up all the associated tests.
      toHash.put("//command_line_option:" + nativeOptionName, value);
    }
    for (String starlarkOptionName : chosenStarlark) {
      Object value =
          toOptions.getStarlarkOptions().get(Label.parseCanonicalUnchecked(starlarkOptionName));
      toHash.put(starlarkOptionName, value);
    }

    if (toHash.isEmpty()) {
      return "";
    } else {
      ImmutableList.Builder<String> hashStrs = ImmutableList.builderWithExpectedSize(toHash.size());
      for (Map.Entry<String, Object> singleOptionAndValue : toHash.entrySet()) {
        Object value = singleOptionAndValue.getValue();
        if (value != null) {
          hashStrs.add(singleOptionAndValue.getKey() + "=" + value);
        } else {
          // Avoid using =null to different from value being the non-null String "null"
          hashStrs.add(singleOptionAndValue.getKey() + "@null");
        }
      }
      return transitionDirectoryNameFragment(hashStrs.build());
    }
  }

  @VisibleForTesting
  public static String transitionDirectoryNameFragment(Iterable<String> opts) {
    Fingerprint fp = new Fingerprint();
    for (String opt : opts) {
      fp.addString(opt);
    }
    // Shorten the hash to 48 bits. This should provide sufficient collision avoidance
    // (that is, we don't expect anyone to experience a collision ever).
    // Shortening the hash is important for Windows paths that tend to be short.
    String suffix = fp.hexDigestAndReset().substring(0, HASH_LENGTH);
    return "ST-" + suffix;
  }
}
