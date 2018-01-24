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
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.common.options.OptionsParsingException;
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

  private final CpuTransformer cpuTransformer;

  /**
   * Creates a new CrosstoolConfigurationLoader instance with the given configuration provider. The
   * configuration provider is used to perform caller-specific configuration file lookup.
   */
  public CppConfigurationLoader(CpuTransformer cpuTransformer) {
    this.cpuTransformer = cpuTransformer;
  }

  @Override
  public CppConfiguration create(ConfigurationEnvironment env, BuildOptions options)
      throws InvalidConfigurationException, InterruptedException {
    CppConfigurationParameters params = createParameters(env, options);
    if (params == null) {
      return null;
    }
    return CppConfiguration.create(params);
  }

  /**
   * Value class for all the data needed to create a {@link CppConfiguration}.
   */
  public static class CppConfigurationParameters {
    protected final CrosstoolConfig.CToolchain toolchain;
    protected final CrosstoolConfigurationLoader.CrosstoolFile crosstoolFile;
    protected final String cacheKeySuffix;
    protected final BuildConfiguration.Options commonOptions;
    protected final CppOptions cppOptions;
    protected final Label crosstoolTop;
    protected final Label ccToolchainLabel;
    protected final Label stlLabel;
    protected final Path fdoZip;
    protected final Label sysrootLabel;
    protected final CpuTransformer cpuTransformer;

    CppConfigurationParameters(
        CrosstoolConfig.CToolchain toolchain,
        CrosstoolConfigurationLoader.CrosstoolFile crosstoolFile,
        String cacheKeySuffix,
        BuildOptions buildOptions,
        Path fdoZip,
        Label crosstoolTop,
        Label ccToolchainLabel,
        Label stlLabel,
        Label sysrootLabel,
        CpuTransformer cpuTransformer) {
      this.toolchain = toolchain;
      this.crosstoolFile = crosstoolFile;
      this.cacheKeySuffix = cacheKeySuffix;
      this.commonOptions = buildOptions.get(BuildConfiguration.Options.class);
      this.cppOptions = buildOptions.get(CppOptions.class);
      this.fdoZip = fdoZip;
      this.crosstoolTop = crosstoolTop;
      this.ccToolchainLabel = ccToolchainLabel;
      this.stlLabel = stlLabel;
      this.sysrootLabel = sysrootLabel;
      this.cpuTransformer = cpuTransformer;
    }
  }

  @Nullable
  protected CppConfigurationParameters createParameters(
      ConfigurationEnvironment env, BuildOptions options)
      throws InvalidConfigurationException, InterruptedException {
    BlazeDirectories directories = env.getBlazeDirectories();
    if (directories == null) {
      return null;
    }
    Label crosstoolTopLabel = RedirectChaser.followRedirects(env,
        options.get(CppOptions.class).crosstoolTop, "crosstool_top");
    if (crosstoolTopLabel == null) {
      return null;
    }

    CppOptions cppOptions = options.get(CppOptions.class);
    Label stlLabel = null;
    if (cppOptions.stl != null) {
      stlLabel = RedirectChaser.followRedirects(env, cppOptions.stl, "stl");
      if (stlLabel == null) {
        return null;
      }
    }

    CrosstoolConfigurationLoader.CrosstoolFile file =
        CrosstoolConfigurationLoader.readCrosstool(env, crosstoolTopLabel);
    if (file == null) {
      return null;
    }
    CrosstoolConfig.CToolchain toolchain =
        CrosstoolConfigurationLoader.selectToolchain(
            file.getProto(), options, cpuTransformer.getTransformer());

    // FDO
    // TODO(bazel-team): move this to CppConfiguration.prepareHook
    Path fdoZip;
    if (cppOptions.getFdoOptimize() == null) {
      fdoZip = null;
    } else if (cppOptions.getFdoOptimize().startsWith("//")) {
      try {
        Target target = env.getTarget(Label.parseAbsolute(cppOptions.getFdoOptimize()));
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
      fdoZip = directories.getWorkspace().getRelative(cppOptions.getFdoOptimize());
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
          .get(toolchain.getTargetCpu() + "|" + toolchain.getCompiler());
      if (ccToolchainLabel == null) {
        throw new InvalidConfigurationException(String.format(
            "cc_toolchain_suite '%s' does not contain a toolchain for CPU '%s' and compiler '%s'",
            crosstoolTopLabel, toolchain.getTargetCpu(), toolchain.getCompiler()));
      }
    } else {
      throw new InvalidConfigurationException(String.format(
          "The specified --crosstool_top '%s' is not a valid cc_toolchain_suite rule",
          crosstoolTopLabel));
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

    if (!(ccToolchain instanceof Rule) || !CcToolchainRule.isCcToolchain(ccToolchain)) {
      throw new InvalidConfigurationException(String.format(
          "The label '%s' is not a cc_toolchain rule", ccToolchainLabel));
    }

    Label sysrootLabel = getSysrootLabel(toolchain, cppOptions.libcTopLabel);

    return new CppConfigurationParameters(
        toolchain,
        file,
        file.getMd5(),
        options,
        fdoZip,
        crosstoolTopLabel,
        ccToolchainLabel,
        stlLabel,
        sysrootLabel,
        cpuTransformer);
  }

  @Nullable
  public static Label getSysrootLabel(CrosstoolConfig.CToolchain toolchain, Label libcTopLabel)
      throws InvalidConfigurationException {
    PathFragment defaultSysroot = CppConfiguration.computeDefaultSysroot(toolchain);

    if ((libcTopLabel != null) && (defaultSysroot == null)) {
      throw new InvalidConfigurationException(
          "The selected toolchain "
              + toolchain.getToolchainIdentifier()
              + " does not support setting --grte_top.");
    }

    if (libcTopLabel != null) {
      return libcTopLabel;
    }

    if (!toolchain.getDefaultGrteTop().isEmpty()) {
      try {
        Label grteTopLabel =
            new CppOptions.LibcTopLabelConverter().convert(toolchain.getDefaultGrteTop());
        return grteTopLabel;
      } catch (OptionsParsingException e) {
        throw new InvalidConfigurationException(e.getMessage(), e);
      }
    }
    return null;
  }
}
