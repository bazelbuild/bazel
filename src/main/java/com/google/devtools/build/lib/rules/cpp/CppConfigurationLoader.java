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

import static com.google.devtools.build.lib.rules.cpp.CrosstoolConfigurationLoader.toReleaseConfiguration;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;
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
    protected final BuildConfiguration.Options commonOptions;
    protected final CppOptions cppOptions;
    protected final Label crosstoolTop;
    protected final Label ccToolchainLabel;
    protected final Label stlLabel;
    protected final PathFragment fdoPath;
    protected final Label fdoOptimizeLabel;
    protected final Label sysrootLabel;
    protected final CcToolchainConfigInfo ccToolchainConfigInfo;
    protected final String transformedCpu;
    protected final String compiler;
    protected final CrosstoolRelease crosstoolFromCcToolchainProtoAttribute;

    CppConfigurationParameters(
        String transformedCpu,
        String compiler,
        CrosstoolRelease crosstoolFromCcToolchainProtoAttribute,
        BuildOptions buildOptions,
        PathFragment fdoPath,
        Label fdoOptimizeLabel,
        Label crosstoolTop,
        Label ccToolchainLabel,
        Label stlLabel,
        Label sysrootLabel,
        CcToolchainConfigInfo ccToolchainConfigInfo) {
      this.transformedCpu = transformedCpu;
      this.compiler = compiler;
      this.crosstoolFromCcToolchainProtoAttribute = crosstoolFromCcToolchainProtoAttribute;
      this.commonOptions = buildOptions.get(BuildConfiguration.Options.class);
      this.cppOptions = buildOptions.get(CppOptions.class);
      this.fdoPath = fdoPath;
      this.fdoOptimizeLabel = fdoOptimizeLabel;
      this.crosstoolTop = crosstoolTop;
      this.ccToolchainLabel = ccToolchainLabel;
      this.stlLabel = stlLabel;
      this.sysrootLabel = sysrootLabel;
      this.ccToolchainConfigInfo = ccToolchainConfigInfo;
    }
  }

  @Nullable
  protected CppConfigurationParameters createParameters(
      ConfigurationEnvironment env, BuildOptions options)
      throws InvalidConfigurationException, InterruptedException {

    CppOptions cppOptions = options.get(CppOptions.class);
    Label crosstoolTopLabel =
        RedirectChaser.followRedirects(env, cppOptions.crosstoolTop, "crosstool_top");
    if (crosstoolTopLabel == null) {
      return null;
    }

    Target crosstoolTop;
    try {
      crosstoolTop = env.getTarget(crosstoolTopLabel);
    } catch (NoSuchThingException e) {
      throw new IllegalStateException(e); // Should have been found out during redirect chasing
    }

    if (!(crosstoolTop instanceof Rule)
        || !((Rule) crosstoolTop).getRuleClass().equals("cc_toolchain_suite")) {
      throw new InvalidConfigurationException(
          String.format(
              "The specified --crosstool_top '%s' is not a valid cc_toolchain_suite rule",
              crosstoolTopLabel));
    }

    CrosstoolRelease crosstoolRelease =
        CrosstoolConfigurationLoader.readCrosstool(env, crosstoolTopLabel);
    if (crosstoolRelease == null) {
      return null;
    }

    Options buildOptions = options.get(Options.class);
    String transformedCpu = cpuTransformer.getTransformer().apply(buildOptions.cpu);
    String key =
        transformedCpu + (cppOptions.cppCompiler == null ? "" : ("|" + cppOptions.cppCompiler));
    Label ccToolchainLabel =
        selectCcToolchainLabel(
            cppOptions,
            crosstoolTopLabel,
            (Rule) crosstoolTop,
            crosstoolRelease,
            transformedCpu,
            key);

    Target ccToolchain = loadCcToolchainTarget(env, ccToolchainLabel);
    if (ccToolchain == null) {
      return null;
    }

    // If cc_toolchain_suite contains an entry for the given --cpu and --compiler options, we
    // select the toolchain by its identifier if "toolchain_identifier" attribute is present.
    // Otherwise, we fall back to going through the CROSSTOOL file to select the toolchain using
    // the legacy selection mechanism.
    String identifierAttribute =
        NonconfigurableAttributeMapper.of((Rule) ccToolchain)
            .get("toolchain_identifier", Type.STRING);
    String cpuAttribute =
        NonconfigurableAttributeMapper.of((Rule) ccToolchain).get("cpu", Type.STRING);
    String compilerAttribute =
        NonconfigurableAttributeMapper.of((Rule) ccToolchain).get("compiler", Type.STRING);

    CToolchain cToolchain =
        CToolchainSelectionUtils.selectCToolchain(
            identifierAttribute,
            cpuAttribute,
            compilerAttribute,
            transformedCpu,
            cppOptions.cppCompiler,
            crosstoolRelease);

    cToolchain =
        CppToolchainInfo.addLegacyFeatures(
            cToolchain, crosstoolTopLabel.getPackageIdentifier().getPathUnderExecRoot());
    CcToolchainConfigInfo ccToolchainConfigInfo = CcToolchainConfigInfo.fromToolchain(cToolchain);

    Label sysrootLabel = getSysrootLabel(cToolchain, cppOptions.libcTopLabel);

    Label stlLabel = null;
    if (cppOptions.stl != null) {
      stlLabel = RedirectChaser.followRedirects(env, cppOptions.stl, "stl");
      if (stlLabel == null) {
        return null;
      }
    }

    String ccToolchainSuiteProtoAttributeValue =
        StringUtil.emptyToNull(
            NonconfigurableAttributeMapper.of((Rule) crosstoolTop).get("proto", Type.STRING));

    CrosstoolRelease crosstoolFromCcToolchainProtoAttribute = null;
    if (ccToolchainSuiteProtoAttributeValue != null) {
      crosstoolFromCcToolchainProtoAttribute =
          toReleaseConfiguration(
              "cc_toolchain_suite rule " + crosstoolTopLabel,
              () -> ccToolchainSuiteProtoAttributeValue,
              /* digestOrNull= */ null);
    }

    PathFragment fdoPath = null;
    Label fdoProfileLabel = null;
    if (cppOptions.getFdoOptimize() != null) {
      if (cppOptions.getFdoOptimize().startsWith("//")) {
        try {
          fdoProfileLabel = Label.parseAbsolute(cppOptions.getFdoOptimize(), ImmutableMap.of());
        } catch (LabelSyntaxException e) {
          throw new InvalidConfigurationException(e);
        }
      } else {
        fdoPath = PathFragment.create(cppOptions.getFdoOptimize());
        try {
          // We don't check for file existence, but at least the filename should be well-formed.
          FileSystemUtils.checkBaseName(fdoPath.getBaseName());
        } catch (IllegalArgumentException e) {
          throw new InvalidConfigurationException(e);
        }
      }
    }

    return new CppConfigurationParameters(
        transformedCpu,
        cppOptions.cppCompiler,
        crosstoolFromCcToolchainProtoAttribute,
        options,
        fdoPath,
        fdoProfileLabel,
        crosstoolTopLabel,
        ccToolchainLabel,
        stlLabel,
        sysrootLabel,
        ccToolchainConfigInfo);
  }

  private Target loadCcToolchainTarget(ConfigurationEnvironment env, Label ccToolchainLabel)
      throws InterruptedException, InvalidConfigurationException {
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
    return ccToolchain;
  }

  private Label selectCcToolchainLabel(
      CppOptions cppOptions,
      Label crosstoolTopLabel,
      Rule crosstoolTop,
      CrosstoolRelease crosstoolRelease,
      String transformedCpu,
      String key)
      throws InvalidConfigurationException {
    String errorMessage =
        String.format(
            "cc_toolchain_suite '%s' does not contain a toolchain for CPU '%s'",
            crosstoolTopLabel, transformedCpu);
    String compiler = cppOptions.cppCompiler;
    if (compiler != null) {
      errorMessage = errorMessage + " and compiler " + compiler;
    }

    Map<String, Label> toolchains =
        NonconfigurableAttributeMapper.of(crosstoolTop)
            .get("toolchains", BuildType.LABEL_DICT_UNARY);
    Label ccToolchainLabel = toolchains.get(key);
    if (ccToolchainLabel == null) {
      // If the cc_toolchain_suite does not contain entry for --cpu|--compiler (or only --cpu if
      // --compiler is not present) we select the toolchain by looping through all the toolchains
      // in the CROSSTOOL file and selecting the one that matches --cpu (and --compiler, if
      // present). Then we use the toolchain.target_cpu|toolchain.compiler key to get the
      // cc_toolchain label.
      CToolchain toolchain =
          CToolchainSelectionUtils.selectCToolchain(
              /* identifierAttribute= */ null,
              /* cpuAttribute= */ null,
              /* compilerAttribute= */ null,
              transformedCpu,
              compiler,
              crosstoolRelease);
      ccToolchainLabel = toolchains.get(toolchain.getTargetCpu() + "|" + toolchain.getCompiler());
      if (cppOptions.disableCcToolchainFromCrosstool) {
        throw new InvalidConfigurationException(
            errorMessage
                + String.format(
                    ", you may want to add an entry for '%s|%s' into toolchains and "
                        + "toolchain_identifier '%s' into the corresponding cc_toolchain rule "
                        + "(see --incompatible_disable_cc_toolchain_label_from_crosstool_proto).",
                    toolchain.getTargetCpu(),
                    toolchain.getCompiler(),
                    toolchain.getToolchainIdentifier()));
      }
    }
    if (ccToolchainLabel == null) {
      throw new InvalidConfigurationException(errorMessage);
    }
    return ccToolchainLabel;
  }

  @Nullable
  public static Label getSysrootLabel(CToolchain toolchain, Label libcTopLabel)
      throws InvalidConfigurationException {
    PathFragment defaultSysroot =
        CppConfiguration.computeDefaultSysroot(toolchain.getBuiltinSysroot());

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
