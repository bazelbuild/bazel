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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;

import javax.annotation.Nullable;

/**
 * Loader for C++ configurations.
 */
public class CppConfigurationLoader implements ConfigurationFragmentFactory {
  @Override
  public Class<? extends Fragment> creates() {
    return CppConfiguration.class;
  }

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
    return ImmutableSet.<Class<? extends FragmentOptions>>of(CppOptions.class);
  }

  private final Function<String, String> cpuTransformer;

  /**
   * Creates a new CrosstoolConfigurationLoader instance with the given
   * configuration provider. The configuration provider is used to perform
   * caller-specific configuration file lookup.
   */
  public CppConfigurationLoader(Function<String, String> cpuTransformer) {
    this.cpuTransformer = cpuTransformer;
  }

  @Override
  public CppConfiguration create(ConfigurationEnvironment env, BuildOptions options)
      throws InvalidConfigurationException {
    CppConfigurationParameters params = createParameters(env, options);
    if (params == null) {
      return null;
    }
    CppConfiguration cppConfig = new CppConfiguration(params);
    if (options.get(BuildConfiguration.Options.class).useDynamicConfigurations
        && cppConfig.getLipoMode() != CrosstoolConfig.LipoMode.OFF) {
      throw new InvalidConfigurationException(
          "LIPO does not currently work with dynamic configurations");
    }
    return cppConfig;
  }

  /**
   * Value class for all the data needed to create a {@link CppConfiguration}.
   */
  public static class CppConfigurationParameters {
    protected final CrosstoolConfig.CToolchain toolchain;
    protected final String cacheKeySuffix;
    protected final BuildConfiguration.Options commonOptions;
    protected final CppOptions cppOptions;
    protected final Label crosstoolTop;
    protected final Label ccToolchainLabel;
    protected final Path fdoZip;
    protected final Path execRoot;

    CppConfigurationParameters(CrosstoolConfig.CToolchain toolchain,
        String cacheKeySuffix,
        BuildOptions buildOptions,
        Path fdoZip,
        Path execRoot,
        Label crosstoolTop,
        Label ccToolchainLabel) {
      this.toolchain = toolchain;
      this.cacheKeySuffix = cacheKeySuffix;
      this.commonOptions = buildOptions.get(BuildConfiguration.Options.class);
      this.cppOptions = buildOptions.get(CppOptions.class);
      this.fdoZip = fdoZip;
      this.execRoot = execRoot;
      this.crosstoolTop = crosstoolTop;
      this.ccToolchainLabel = ccToolchainLabel;
    }
  }

  @Nullable
  protected CppConfigurationParameters createParameters(
      ConfigurationEnvironment env, BuildOptions options) throws InvalidConfigurationException {
    BlazeDirectories directories = env.getBlazeDirectories();
    if (directories == null) {
      return null;
    }
    Label crosstoolTopLabel = RedirectChaser.followRedirects(env,
        options.get(CppOptions.class).crosstoolTop, "crosstool_top");
    if (crosstoolTopLabel == null) {
      return null;
    }

    CrosstoolConfigurationLoader.CrosstoolFile file =
        CrosstoolConfigurationLoader.readCrosstool(env, crosstoolTopLabel);
    if (file == null) {
      return null;
    }
    CrosstoolConfig.CToolchain toolchain =
        CrosstoolConfigurationLoader.selectToolchain(file.getProto(), options, cpuTransformer);

    // FDO
    // TODO(bazel-team): move this to CppConfiguration.prepareHook
    CppOptions cppOptions = options.get(CppOptions.class);
    Path fdoZip;
    if (cppOptions.fdoOptimize == null) {
      fdoZip = null;
    } else if (cppOptions.fdoOptimize.startsWith("//")) {
      try {
        Target target = env.getTarget(Label.parseAbsolute(cppOptions.fdoOptimize));
        if (target == null) {
          return null;
        }
        if (!(target instanceof InputFile)) {
          throw new InvalidConfigurationException(
              "--fdo_optimize cannot accept targets that do not refer to input files");
        }
        fdoZip = env.getPath(target.getPackage(), target.getName());
        if (fdoZip == null) {
          throw new InvalidConfigurationException(
              "The --fdo_optimize parameter you specified resolves to a file that does not exist");
        }
      } catch (NoSuchPackageException | NoSuchTargetException | LabelSyntaxException e) {
        env.getEventHandler().handle(Event.error(e.getMessage()));
        throw new InvalidConfigurationException(e);
      }
    } else {
      fdoZip = directories.getWorkspace().getRelative(cppOptions.fdoOptimize);
      try {
        // We don't check for file existence, but at least the filename should be well-formed.
        FileSystemUtils.checkBaseName(fdoZip.getBaseName());
      } catch (IllegalArgumentException e) {
        throw new InvalidConfigurationException(e);
      }
    }

    Label ccToolchainLabel;
    Target crosstoolTop;

    try {
      crosstoolTop = env.getTarget(crosstoolTopLabel);
    } catch (NoSuchThingException e) {
      throw new IllegalStateException(e);  // Should have been found out during redirect chasing
    }

    if (crosstoolTop instanceof Rule
        && ((Rule) crosstoolTop).getRuleClass().equals("cc_toolchain_suite")) {
      Rule ccToolchainSuite = (Rule) crosstoolTop;
      ccToolchainLabel = NonconfigurableAttributeMapper.of(ccToolchainSuite)
          .get("toolchains", BuildType.LABEL_DICT_UNARY)
          .get(toolchain.getTargetCpu());
      if (ccToolchainLabel == null) {
        throw new InvalidConfigurationException(String.format(
            "cc_toolchain_suite '%s' does not contain a toolchain for CPU '%s'",
            crosstoolTopLabel, toolchain.getTargetCpu()));
      }
    } else {
      try {
        ccToolchainLabel = crosstoolTopLabel.getRelative("cc-compiler-" + toolchain.getTargetCpu());
      } catch (LabelSyntaxException e) {
        throw new InvalidConfigurationException(String.format(
            "'%s' is not a valid CPU. It should only consist of characters valid in labels",
            toolchain.getTargetCpu()));
      }
    }

    Target ccToolchain;
    try {
      ccToolchain = env.getTarget(ccToolchainLabel);
      if (ccToolchain == null) {
        return null;
      }
    } catch (NoSuchThingException e) {
      throw new InvalidConfigurationException(String.format(
          "The toolchain rule '%s' does not exist", ccToolchainLabel));
    }

    if (!(ccToolchain instanceof Rule)
        || !((Rule) ccToolchain).getRuleClass().equals("cc_toolchain")) {
      throw new InvalidConfigurationException(String.format(
          "The label '%s' is not a cc_toolchain rule", ccToolchainLabel));
    }

    return new CppConfigurationParameters(toolchain, file.getMd5(), options,
        fdoZip, directories.getExecRoot(), crosstoolTopLabel, ccToolchainLabel);
  }
}
