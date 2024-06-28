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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.rules.cpp.LinkBuildVariables.LINKER_PARAM_FILE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkerOrArchiver;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Represents the command line of a linker invocation. It supports executables and dynamic libraries
 * as well as static libraries.
 */
@Immutable
public final class LinkCommandLine extends CommandLine {
  private final String actionName;
  private final String forcedToolPath;
  private final CcToolchainVariables variables;
  // The feature config can be null for tests.
  @Nullable private final FeatureConfiguration featureConfiguration;
  private final ImmutableList<Artifact> buildInfoHeaderArtifacts;
  private final NestedSet<Artifact> linkerInputArtifacts;
  private final LinkTargetType linkTargetType;
  private final Link.LinkingMode linkingMode;
  @Nullable private final PathFragment toolchainLibrariesSolibDir;
  private final boolean nativeDeps;
  private final boolean useTestOnlyFlags;

  @Nullable private final Artifact paramFile;

  private LinkCommandLine(
      String actionName,
      String forcedToolPath,
      ImmutableList<Artifact> buildInfoHeaderArtifacts,
      NestedSet<Artifact> linkerInputArtifacts,
      LinkTargetType linkTargetType,
      Link.LinkingMode linkingMode,
      @Nullable PathFragment toolchainLibrariesSolibDir,
      boolean nativeDeps,
      boolean useTestOnlyFlags,
      @Nullable Artifact paramFile,
      CcToolchainVariables variables,
      @Nullable FeatureConfiguration featureConfiguration) {

    this.actionName = actionName;
    this.forcedToolPath = forcedToolPath;
    this.variables = variables;
    this.featureConfiguration = featureConfiguration;
    this.buildInfoHeaderArtifacts = Preconditions.checkNotNull(buildInfoHeaderArtifacts);
    this.linkerInputArtifacts = Preconditions.checkNotNull(linkerInputArtifacts);
    this.linkTargetType = Preconditions.checkNotNull(linkTargetType);
    this.linkingMode = Preconditions.checkNotNull(linkingMode);
    this.toolchainLibrariesSolibDir = toolchainLibrariesSolibDir;
    this.nativeDeps = nativeDeps;
    this.useTestOnlyFlags = useTestOnlyFlags;
    this.paramFile = paramFile;
  }

  @Nullable
  public Artifact getParamFile() {
    return paramFile;
  }

  /** See {@link CppLinkAction#getBuildInfoHeaderArtifacts()} */
  public ImmutableList<Artifact> getBuildInfoHeaderArtifacts() {
    return buildInfoHeaderArtifacts;
  }

  /** Returns the (ordered, immutable) list of paths to the linker's input files. */
  public NestedSet<Artifact> getLinkerInputArtifacts() {
    return linkerInputArtifacts;
  }

  @Nullable
  @VisibleForTesting
  public FeatureConfiguration getFeatureConfiguration() {
    return featureConfiguration;
  }

  public String getActionName() {
    return actionName;
  }

  /** Returns the current type of link target set. */
  public LinkTargetType getLinkTargetType() {
    return linkTargetType;
  }

  /** Returns the "staticness" of the link. */
  public Link.LinkingMode getLinkingMode() {
    return linkingMode;
  }

  /** Returns the path to the linker. */
  public String getLinkerPathString() {
    if (forcedToolPath != null) {
      return forcedToolPath;
    } else {
      Preconditions.checkArgument(
          featureConfiguration.actionIsConfigured(actionName),
          "Expected action_config for '%s' to be configured",
          actionName);
      return featureConfiguration.getToolPathForAction(linkTargetType.getActionName());
    }
  }

  /**
   * Returns the location of the C++ runtime solib symlinks. If null, the C++ dynamic runtime
   * libraries either do not exist (because they do not come from the depot) or they are in the
   * regular solib directory.
   */
  @Nullable
  public PathFragment getToolchainLibrariesSolibDir() {
    return toolchainLibrariesSolibDir;
  }

  /** Returns true for libraries linked as native dependencies for other languages. */
  public boolean isNativeDeps() {
    return nativeDeps;
  }

  /**
   * Returns true if this link should use test-specific flags (e.g. $EXEC_ORIGIN as the root for
   * finding shared libraries or lazy binding); false by default. See bug "Please use $EXEC_ORIGIN
   * instead of $ORIGIN when linking cc_tests" for further context.
   */
  public boolean useTestOnlyFlags() {
    return useTestOnlyFlags;
  }

  /** Returns the build variables used to template the crosstool for this linker invocation. */
  @VisibleForTesting
  public CcToolchainVariables getBuildVariables() {
    return this.variables;
  }

  /** Returns just the .params file portion of the command-line as a {@link CommandLine}. */
  CommandLine paramCmdLine() {
    Preconditions.checkNotNull(paramFile);
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() throws CommandLineExpansionException {
        return getParamCommandLine(null);
      }

      @Override
      public List<String> arguments(ArtifactExpander expander, PathMapper pathMapper)
          throws CommandLineExpansionException {
        return getParamCommandLine(expander);
      }
    };
  }

  public List<String> getCommandLine(@Nullable ArtifactExpander expander)
      throws CommandLineExpansionException {
    // Try to shorten the command line by use of a parameter file.
    // This makes the output with --subcommands (et al) more readable.
    List<String> argv = new ArrayList<>();
    argv.add(getLinkerPathString());
    try {
      if (paramFile != null) {
        // Retrieve only reference to linker_param_file from the command line.
        String linkerParamFile =
            variables
                .getVariable(LINKER_PARAM_FILE.getVariableName())
                .getStringValue(LINKER_PARAM_FILE.getVariableName(), PathMapper.NOOP);
        argv.addAll(
            featureConfiguration
                .getCommandLine(actionName, variables, expander, PathMapper.NOOP)
                .stream()
                .filter(s -> s.contains(linkerParamFile))
                .collect(toImmutableList()));
      } else {
        argv.addAll(
            featureConfiguration.getCommandLine(actionName, variables, expander, PathMapper.NOOP));
      }
    } catch (ExpansionException e) {
      throw new CommandLineExpansionException(e.getMessage());
    }
    return argv;
  }

  public List<String> getParamCommandLine(@Nullable ArtifactExpander expander)
      throws CommandLineExpansionException {
    List<String> argv = new ArrayList<>();
    try {
      if (variables.isAvailable(LINKER_PARAM_FILE.getVariableName())) {
        // Filter out linker_param_file
        String linkerParamFile =
            variables
                .getVariable(LINKER_PARAM_FILE.getVariableName())
                .getStringValue(LINKER_PARAM_FILE.getVariableName(), PathMapper.NOOP);
        argv.addAll(
            featureConfiguration
                .getCommandLine(actionName, variables, expander, PathMapper.NOOP)
                .stream()
                .filter(s -> !s.contains(linkerParamFile))
                .collect(toImmutableList()));
      } else {
        argv.addAll(
            featureConfiguration.getCommandLine(actionName, variables, expander, PathMapper.NOOP));
      }
    } catch (ExpansionException e) {
      throw new CommandLineExpansionException(e.getMessage());
    }
    return argv;
  }

  @Override
  public List<String> arguments() throws CommandLineExpansionException {
    return arguments(null, null);
  }

  @Override
  public List<String> arguments(ArtifactExpander artifactExpander, PathMapper pathMapper)
      throws CommandLineExpansionException {
    return ImmutableList.<String>builder()
        .add(getLinkerPathString())
        .addAll(getParamCommandLine(artifactExpander))
        .build();
  }

  /** A builder for a {@link LinkCommandLine}. */
  public static final class Builder {

    private String forcedToolPath;
    private ImmutableList<Artifact> buildInfoHeaderArtifacts = ImmutableList.of();
    private NestedSet<Artifact> linkerInputArtifacts =
        NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    @Nullable private LinkTargetType linkTargetType;
    private Link.LinkingMode linkingMode = Link.LinkingMode.STATIC;
    @Nullable private PathFragment toolchainLibrariesSolibDir;
    private boolean nativeDeps;
    private boolean useTestOnlyFlags;
    @Nullable private Artifact paramFile;
    private CcToolchainVariables variables;
    private FeatureConfiguration featureConfiguration;
    private String actionName;

    public LinkCommandLine build() {
      if (linkTargetType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER) {
        Preconditions.checkArgument(
            buildInfoHeaderArtifacts.isEmpty(),
            "build info headers may only be present on dynamic library or executable links");
      }

      if (variables == null) {
        variables = CcToolchainVariables.EMPTY;
      }

      return new LinkCommandLine(
          actionName,
          forcedToolPath,
          buildInfoHeaderArtifacts,
          linkerInputArtifacts,
          linkTargetType,
          linkingMode,
          toolchainLibrariesSolibDir,
          nativeDeps,
          useTestOnlyFlags,
          paramFile,
          variables,
          featureConfiguration);
    }

    /** Use given tool path instead of the one from feature configuration */
    @CanIgnoreReturnValue
    public Builder forceToolPath(String forcedToolPath) {
      this.forcedToolPath = forcedToolPath;
      return this;
    }

    /** Sets the feature configuration for this link action. */
    @CanIgnoreReturnValue
    public Builder setFeatureConfiguration(FeatureConfiguration featureConfiguration) {
      this.featureConfiguration = featureConfiguration;
      return this;
    }

    /**
     * Sets the type of the link. It is an error to try to set this to {@link
     * LinkTargetType#INTERFACE_DYNAMIC_LIBRARY}. Note that all the static target types (see {@link
     * LinkTargetType#linkerOrArchiver}) are equivalent, and there is no check that the output
     * artifact matches the target type extension.
     */
    @CanIgnoreReturnValue
    public Builder setLinkTargetType(LinkTargetType linkTargetType) {
      Preconditions.checkArgument(linkTargetType != LinkTargetType.INTERFACE_DYNAMIC_LIBRARY);
      this.linkTargetType = linkTargetType;
      return this;
    }

    /**
     * Sets a list of linker input artifacts. These get turned into linker options depending on the
     * staticness and the target type. This call makes an immutable copy of the inputs, if the
     * provided Iterable isn't already immutable (see {@link CollectionUtils#makeImmutable}).
     */
    @CanIgnoreReturnValue
    public Builder setLinkerInputArtifacts(NestedSet<Artifact> linkerInputArtifacts) {
      this.linkerInputArtifacts = linkerInputArtifacts;
      return this;
    }

    /**
     * Sets how static the link is supposed to be. For static target types (see {@link
     * LinkTargetType#linkerOrArchiver()}}), the {@link #build} method throws an exception if this
     * is not {@link LinkingMode#STATIC}. The default setting is {@link LinkingMode#STATIC}.
     */
    @CanIgnoreReturnValue
    public Builder setLinkingMode(Link.LinkingMode linkingMode) {
      this.linkingMode = linkingMode;
      return this;
    }

    /**
     * The build info header artifacts are generated header files that are used for link stamping.
     * The {@link #build} method throws an exception if the build info header artifacts are
     * non-empty for a static link (see {@link LinkTargetType#linkerOrArchiver()}}).
     */
    @CanIgnoreReturnValue
    public Builder setBuildInfoHeaderArtifacts(ImmutableList<Artifact> buildInfoHeaderArtifacts) {
      this.buildInfoHeaderArtifacts = buildInfoHeaderArtifacts;
      return this;
    }

    /**
     * Whether the resulting library is intended to be used as a native library from another
     * programming language. This influences the rpath. The {@link #build} method throws an
     * exception if this is true for a static link (see {@link LinkTargetType#linkerOrArchiver()}}).
     */
    @CanIgnoreReturnValue
    public Builder setNativeDeps(boolean nativeDeps) {
      this.nativeDeps = nativeDeps;
      return this;
    }

    /**
     * Sets whether to use test-specific linker flags, e.g. {@code $EXEC_ORIGIN} instead of {@code
     * $ORIGIN} in the rpath or lazy binding.
     */
    @CanIgnoreReturnValue
    public Builder setUseTestOnlyFlags(boolean useTestOnlyFlags) {
      this.useTestOnlyFlags = useTestOnlyFlags;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setParamFile(Artifact paramFile) {
      this.paramFile = paramFile;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setBuildVariables(CcToolchainVariables variables) {
      this.variables = variables;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setToolchainLibrariesSolibDir(PathFragment toolchainLibrariesSolibDir) {
      this.toolchainLibrariesSolibDir = toolchainLibrariesSolibDir;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setActionName(String actionName) {
      this.actionName = actionName;
      return this;
    }
  }
}
