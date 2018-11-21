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
// limitations under the License.
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import java.util.ArrayList;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Utils class for logic responsible for selecting a CToolchain from the CROSSTOOL file given the
 * cc_toolchain.
 */
public class CToolchainSelectionUtils {

  /**
   * Do-it-all CToolchain selection method that considers toolchain identifiers, cpu/compiler
   * attributes, and cpu/compiler options. Returns the CToolchain from the CROSSTOOL. If you need to
   * call anything else than this, you're holding it wrong.
   *
   * @param identifierAttribute value of the cc_toolchain.toolchain_identifier attribute
   * @param cpuAttribute value of the cc_toolchain.cpu attribute
   * @param compilerAttribute value of the cc_toolchain.compiler attribute
   * @param cpuOption value of the --cpu option
   * @param compilerOption value of the --compiler option
   * @param proto content of the CROSSTOOL file
   * @return selected CToolchain or throws InvalidConfigurationException when not found. Never
   *     returns null.
   */
  static CToolchain selectCToolchain(
      @Nullable String identifierAttribute,
      @Nullable String cpuAttribute,
      @Nullable String compilerAttribute,
      String cpuOption,
      @Nullable String compilerOption,
      CrosstoolRelease proto)
      throws InvalidConfigurationException {
    String identifierAttributeOrNull = StringUtil.emptyToNull(identifierAttribute);
    String cpuAttributeOrNull = StringUtil.emptyToNull(cpuAttribute);
    String compilerAttributeOrNull = StringUtil.emptyToNull(compilerAttribute);

    Preconditions.checkNotNull(cpuOption);
    String compilerOptionOrNull = StringUtil.emptyToNull(compilerOption);

    return selectCToolchainNoEmptyStrings(
        identifierAttributeOrNull,
        cpuAttributeOrNull,
        compilerAttributeOrNull,
        cpuOption,
        compilerOptionOrNull,
        proto);
  }

  private static CToolchain selectCToolchainNoEmptyStrings(
      String identifierAttribute,
      String cpuAttribute,
      String compilerAttribute,
      String cpuOption,
      String compilerOption,
      CrosstoolRelease proto)
      throws InvalidConfigurationException {
    CToolchain cToolchain = null;
    // Use the identifier to find the CToolchain from the CROSSTOOL (this is the way how
    // cc_toolchain will select CToolchain in the happy future, since it works with platforms).
    if (identifierAttribute != null) {
      cToolchain = getToolchainByIdentifier(proto, identifierAttribute);
    }
    if (cToolchain == null && cpuAttribute != null) {
      // Let's try to select the CToolchain using cpu and compiler rule attributes (the semi-new
      // way, compatible with platforms).
      try {
        cToolchain =
            selectToolchainUsingCpuAndMaybeCompiler(
                proto, new CrosstoolConfigurationIdentifier(cpuAttribute, compilerAttribute));
      } catch (InvalidConfigurationException e) {
        // We couldn't find the CToolchain using attributes, let's catch the exception and try
        // with options. It's safe to ignore the exception here, since if it was caused by
        // something else than the selection, it will be re-thrown below.
      }
    }
    if (cToolchain == null) {
      // We couldn't find the CToolchain using cpu and compiler attributes, let's try to select
      // it using --cpu/--compiler options (the legacy way, doesn't work with platforms).
      cToolchain =
          selectToolchainUsingCpuAndMaybeCompiler(
              proto, new CrosstoolConfigurationIdentifier(cpuOption, compilerOption));
    }
    return cToolchain;
  }

  /**
   * Selects a crosstool toolchain based on the toolchain identifier.
   *
   * @throws InvalidConfigurationException if no matching toolchain can be found, or if multiple
   *     toolchains with the same identifier are found.
   */
  private static CToolchain getToolchainByIdentifier(
      CrosstoolRelease proto, String toolchainIdentifier) throws InvalidConfigurationException {
    checkToolchain(toolchainIdentifier);
    CToolchain selectedToolchain = null;
    for (CToolchain toolchain : proto.getToolchainList()) {
      if (toolchain.getToolchainIdentifier().equals(toolchainIdentifier)) {
        if (selectedToolchain != null) {
          throw new InvalidConfigurationException(
              String.format("Multiple toolchains with '%s' identifier", toolchainIdentifier));
        }
        selectedToolchain = toolchain;
      }
    }
    if (selectedToolchain == null) {
      throw new InvalidConfigurationException(
          String.format(
                  "Toolchain identifier '%s' was not found, valid identifiers are %s",
                  toolchainIdentifier,
                  proto.getToolchainList().stream()
                      .map(CToolchain::getToolchainIdentifier)
                      .collect(ImmutableList.toImmutableList())));
    }
    return selectedToolchain;
  }

  /**
   * Makes sure that {@code selectedIdentifier} is a valid identifier for a toolchain, i.e. it
   * starts with a letter or an underscore and continues with only dots, dashes, spaces, letters,
   * digits or underscores (i.e. matches the following regular expression: "[a-zA-Z_][\.\- \w]*").
   *
   * @throws InvalidConfigurationException if selectedIdentifier does not match the aforementioned
   *     regular expression.
   */
  private static void checkToolchain(String selectedIdentifier)
      throws InvalidConfigurationException {
    // If you update this regex, please do so in the javadoc comment too, and also in the
    // crosstool_config.proto file.
    String rx = "[a-zA-Z_][\\.\\- \\w]*";
    if (!selectedIdentifier.matches(rx)) {
      throw new InvalidConfigurationException(
          String.format(
              "Toolchain identifier '%s' is illegal (does not match '%s')",
              selectedIdentifier, rx));
    }
  }

  /**
   * Selects a crosstool toolchain corresponding to the given crosstool configuration options. If
   * all of these options are null, it returns the default toolchain specified in the crosstool
   * release. If only cpu is non-null, it returns the default toolchain for that cpu, as specified
   * in the crosstool release. Otherwise, all values must be non-null, and this method returns the
   * toolchain which matches all of the values.
   *
   * @throws NullPointerException if {@code release} is null
   * @throws InvalidConfigurationException if no matching toolchain can be found, or if the input
   *     parameters do not obey the constraints described above
   */
  private static CToolchain selectToolchainUsingCpuAndMaybeCompiler(
      CrosstoolRelease release, CrosstoolConfigurationIdentifier config)
      throws InvalidConfigurationException {
    if (config.getCompiler() != null) {
      ArrayList<CToolchain> candidateToolchains = new ArrayList<>();
      for (CToolchain toolchain : release.getToolchainList()) {
        if (config.isCandidateToolchain(toolchain)) {
          candidateToolchains.add(toolchain);
        }
      }
      switch (candidateToolchains.size()) {
        case 0:
          {
            StringBuilder message = new StringBuilder();
            message.append("No toolchain found for");
            message.append(config.describeFlags());
            message.append(". Valid toolchains are: ");
            describeToolchainList(message, release.getToolchainList());
            throw new InvalidConfigurationException(message.toString());
          }
        case 1:
          return candidateToolchains.get(0);
        default:
          {
            StringBuilder message = new StringBuilder();
            message.append("Multiple toolchains found for");
            message.append(config.describeFlags());
            message.append(": ");
            describeToolchainList(message, candidateToolchains);
            throw new InvalidConfigurationException(message.toString());
          }
      }
    }

    StringBuilder errorMessageBuilder = new StringBuilder();
    errorMessageBuilder
        .append("No toolchain found for cpu '")
        .append(config.getCpu())
        .append("'. Valid toolchains are: ");
    describeToolchainList(errorMessageBuilder, release.getToolchainList());
    throw new InvalidConfigurationException(errorMessageBuilder.toString());
  }

  /**
   * Appends a series of toolchain descriptions (as the blaze command line flags that would specify
   * that toolchain) to 'message'.
   */
  private static void describeToolchainList(
      StringBuilder message, Collection<CToolchain> toolchains) {
    message.append("[\n");
    for (CrosstoolConfig.CToolchain toolchain : toolchains) {
      message.append("  ");
      message.append(toolchain.getToolchainIdentifier());
      message.append(": ");
      message.append(
          CrosstoolConfigurationIdentifier.fromToolchain(toolchain).describeFlags().trim());
      message.append(",\n");
    }
    message.append("]");
  }
}
