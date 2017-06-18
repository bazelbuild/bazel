// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction.DotdFile;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** The compile command line for the C++ compile action. */
public final class CompileCommandLine {

  private final Artifact sourceFile;
  private final Artifact outputFile;
  private final Label sourceLabel;
  private final List<String> copts;
  private final Predicate<String> coptsFilter;
  private final Collection<String> features;
  private final FeatureConfiguration featureConfiguration;
  private final CcToolchainFeatures.Variables variables;
  private final String actionName;
  private final CppConfiguration cppConfiguration;
  private final CcToolchainProvider cppProvider;
  private final DotdFile dotdFile;

  private CompileCommandLine(
      Artifact sourceFile,
      Artifact outputFile,
      Label sourceLabel,
      ImmutableList<String> copts,
      Predicate<String> coptsFilter,
      Collection<String> features,
      FeatureConfiguration featureConfiguration,
      CppConfiguration cppConfiguration,
      CcToolchainFeatures.Variables variables,
      String actionName,
      DotdFile dotdFile,
      CcToolchainProvider cppProvider) {
    this.sourceFile = Preconditions.checkNotNull(sourceFile);
    this.outputFile = Preconditions.checkNotNull(outputFile);
    this.sourceLabel = Preconditions.checkNotNull(sourceLabel);
    this.copts = Preconditions.checkNotNull(copts);
    this.coptsFilter = coptsFilter;
    this.features = Preconditions.checkNotNull(features);
    this.featureConfiguration = Preconditions.checkNotNull(featureConfiguration);
    this.cppConfiguration = Preconditions.checkNotNull(cppConfiguration);
    this.variables = variables;
    this.actionName = actionName;
    this.dotdFile = isGenerateDotdFile(sourceFile) ? Preconditions.checkNotNull(dotdFile) : null;
    this.cppProvider = cppProvider;
  }

  /** Returns true if Dotd file should be generated. */
  private boolean isGenerateDotdFile(Artifact sourceArtifact) {
    return CppFileTypes.headerDiscoveryRequired(sourceArtifact)
        && !featureConfiguration.isEnabled(CppRuleClasses.PARSE_SHOWINCLUDES);
  }

  /** Returns the environment variables that should be set for C++ compile actions. */
  protected Map<String, String> getEnvironment() {
    return featureConfiguration.getEnvironmentVariables(actionName, variables);
  }

  protected List<String> getArgv(
      PathFragment outputFile, CcToolchainFeatures.Variables overwrittenVariables) {
    List<String> commandLine = new ArrayList<>();

    // first: The command name.
    Preconditions.checkArgument(
        featureConfiguration.actionIsConfigured(actionName),
        String.format("Expected action_config for '%s' to be configured", actionName));
    commandLine.add(
        featureConfiguration
            .getToolForAction(actionName)
            .getToolPath(cppConfiguration.getCrosstoolTopPathFragment())
            .getPathString());

    // second: The compiler options.
    commandLine.addAll(getCompilerOptions(overwrittenVariables));

    if (!featureConfiguration.isEnabled("compile_action_flags_in_flag_set")) {
      // third: The file to compile!
      commandLine.add("-c");
      commandLine.add(sourceFile.getExecPathString());

      // finally: The output file. (Prefixed with -o).
      commandLine.add("-o");
      commandLine.add(outputFile.getPathString());
    }

    return commandLine;
  }

  private boolean isObjcCompile(String actionName) {
    return (actionName.equals(CppCompileAction.OBJC_COMPILE)
        || actionName.equals(CppCompileAction.OBJCPP_COMPILE));
  }

  public List<String> getCompilerOptions(
      @Nullable CcToolchainFeatures.Variables overwrittenVariables) {
    List<String> options = new ArrayList<>();
    CppConfiguration toolchain = cppConfiguration;

    addFilteredOptions(options, toolchain.getCompilerOptions(features));

    String sourceFilename = sourceFile.getExecPathString();
    if (CppFileTypes.C_SOURCE.matches(sourceFilename)) {
      addFilteredOptions(options, toolchain.getCOptions());
    }
    if (CppFileTypes.CPP_SOURCE.matches(sourceFilename)
        || CppFileTypes.CPP_HEADER.matches(sourceFilename)
        || CppFileTypes.CPP_MODULE_MAP.matches(sourceFilename)
        || CppFileTypes.CLIF_INPUT_PROTO.matches(sourceFilename)) {
      addFilteredOptions(options, toolchain.getCxxOptions(features));
    }

    // TODO(bazel-team): This needs to be before adding getUnfilteredCompilerOptions() and after
    // adding the warning flags until all toolchains are migrated; currently toolchains use the
    // unfiltered compiler options to inject include paths, which is superseded by the feature
    // configuration; on the other hand toolchains switch off warnings for the layering check
    // that will be re-added by the feature flags.
    CcToolchainFeatures.Variables updatedVariables = variables;
    if (variables != null && overwrittenVariables != null) {
      CcToolchainFeatures.Variables.Builder variablesBuilder =
          new CcToolchainFeatures.Variables.Builder();
      variablesBuilder.addAll(variables);
      variablesBuilder.addAndOverwriteAll(overwrittenVariables);
      updatedVariables = variablesBuilder.build();
    }
    addFilteredOptions(options, featureConfiguration.getCommandLine(actionName, updatedVariables));

    // Users don't expect the explicit copts to be filtered by coptsFilter, add them verbatim.
    // Make sure these are added after the options from the feature configuration, so that
    // those options can be overriden.
    options.addAll(copts);

    // Unfiltered compiler options contain system include paths. These must be added after
    // the user provided options, otherwise users adding include paths will not pick up their
    // own include paths first.
    if (!isObjcCompile(actionName)) {
      options.addAll(cppProvider.getUnfilteredCompilerOptions(features));
    }

    // Add the options of --per_file_copt, if the label or the base name of the source file
    // matches the specified regular expression filter.
    for (PerLabelOptions perLabelOptions : cppConfiguration.getPerFileCopts()) {
      if ((sourceLabel != null && perLabelOptions.isIncluded(sourceLabel))
          || perLabelOptions.isIncluded(sourceFile)) {
        options.addAll(perLabelOptions.getOptions());
      }
    }

    if (!featureConfiguration.isEnabled("compile_action_flags_in_flag_set")) {
      if (FileType.contains(outputFile, CppFileTypes.ASSEMBLER, CppFileTypes.PIC_ASSEMBLER)) {
        options.add("-S");
      } else if (FileType.contains(
          outputFile,
          CppFileTypes.PREPROCESSED_C,
          CppFileTypes.PREPROCESSED_CPP,
          CppFileTypes.PIC_PREPROCESSED_C,
          CppFileTypes.PIC_PREPROCESSED_CPP)) {
        options.add("-E");
      }
    }

    return options;
  }

  // For each option in 'in', add it to 'out' unless it is matched by the 'coptsFilter' regexp.
  private void addFilteredOptions(List<String> out, List<String> in) {
    Iterables.addAll(out, Iterables.filter(in, coptsFilter));
  }

  public Artifact getSourceFile() {
    return sourceFile;
  }

  public DotdFile getDotdFile() {
    return dotdFile;
  }

  public List<String> getCopts() {
    return copts;
  }

  public Variables getVariables() {
    return variables;
  }

  public static Builder builder(
      Artifact sourceFile,
      Artifact outputFile,
      Label sourceLabel,
      ImmutableList<String> copts,
      Predicate<String> coptsFilter,
      ImmutableList<String> features,
      String actionName,
      CppConfiguration cppConfiguration,
      DotdFile dotdFile,
      CcToolchainProvider cppProvider) {
    return new Builder(
        sourceFile,
        outputFile,
        sourceLabel,
        copts,
        coptsFilter,
        features,
        actionName,
        cppConfiguration,
        dotdFile,
        cppProvider);
  }

  /** A builder for a {@link CompileCommandLine}. */
  public static final class Builder {
    private final Artifact sourceFile;
    private final Artifact outputFile;
    private final Label sourceLabel;
    private final ImmutableList<String> copts;
    private final Predicate<String> coptsFilter;
    private final Collection<String> features;
    private FeatureConfiguration featureConfiguration;
    private CcToolchainFeatures.Variables variables = Variables.EMPTY;
    private final String actionName;
    private final CppConfiguration cppConfiguration;
    @Nullable private final DotdFile dotdFile;
    private final CcToolchainProvider ccToolchainProvider;

    public CompileCommandLine build() {
      return new CompileCommandLine(
          Preconditions.checkNotNull(sourceFile),
          Preconditions.checkNotNull(outputFile),
          Preconditions.checkNotNull(sourceLabel),
          Preconditions.checkNotNull(copts),
          Preconditions.checkNotNull(coptsFilter),
          Preconditions.checkNotNull(features),
          Preconditions.checkNotNull(featureConfiguration),
          Preconditions.checkNotNull(cppConfiguration),
          Preconditions.checkNotNull(variables),
          Preconditions.checkNotNull(actionName),
          dotdFile,
          Preconditions.checkNotNull(ccToolchainProvider));
    }

    private Builder(
        Artifact sourceFile,
        Artifact outputFile,
        Label sourceLabel,
        ImmutableList<String> copts,
        Predicate<String> coptsFilter,
        Collection<String> features,
        String actionName,
        CppConfiguration cppConfiguration,
        DotdFile dotdFile,
        CcToolchainProvider ccToolchainProvider) {
      this.sourceFile = sourceFile;
      this.outputFile = outputFile;
      this.sourceLabel = sourceLabel;
      this.copts = copts;
      this.coptsFilter = coptsFilter;
      this.features = features;
      this.actionName = actionName;
      this.cppConfiguration = cppConfiguration;
      this.dotdFile = dotdFile;
      this.ccToolchainProvider = ccToolchainProvider;
    }

    /** Sets the feature configuration for this compile action. */
    public Builder setFeatureConfiguration(FeatureConfiguration featureConfiguration) {
      this.featureConfiguration = featureConfiguration;
      return this;
    }

    public Builder setVariables(Variables variables) {
      this.variables = variables;
      return this;
    }
  }
}
