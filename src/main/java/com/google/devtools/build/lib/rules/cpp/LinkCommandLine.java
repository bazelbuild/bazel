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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkerOrArchiver;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Represents the command line of a linker invocation. It supports executables and dynamic libraries
 * as well as static libraries.
 */
@AutoCodec
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

  @VisibleForSerialization
  LinkCommandLine(
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
    return featureConfiguration.getToolPathForAction(linkTargetType.getActionName());
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

  /**
   * Splits the link command-line into a part to be written to a parameter file, and the remaining
   * actual command line to be executed (which references the parameter file). Should only be used
   * if getParamFile() is not null.
   */
  @VisibleForTesting
  final Pair<List<String>, List<String>> splitCommandline() throws CommandLineExpansionException {
    return splitCommandline(paramFile, getRawLinkArgv(null), linkTargetType);
  }

  @VisibleForTesting
  final Pair<List<String>, List<String>> splitCommandline(@Nullable ArtifactExpander expander)
      throws CommandLineExpansionException {
    return splitCommandline(paramFile, getRawLinkArgv(expander), linkTargetType);
  }

  private static Pair<List<String>, List<String>> splitCommandline(
      Artifact paramFile, List<String> args, LinkTargetType linkTargetType) {
    Preconditions.checkNotNull(paramFile);
    if (linkTargetType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER) {
      // Ar link commands can also generate huge command lines.
      List<String> paramFileArgs = new ArrayList<>();
      List<String> commandlineArgs = new ArrayList<>();
      extractArgumentsForStaticLinkParamFile(args, commandlineArgs, paramFileArgs);
      return Pair.of(commandlineArgs, paramFileArgs);
    } else {
      // Gcc link commands tend to generate humongous commandlines for some targets, which may
      // not fit on some remote execution machines. To work around this we will employ the help of
      // a parameter file and pass any linker options through it.
      List<String> paramFileArgs = new ArrayList<>();
      List<String> commandlineArgs = new ArrayList<>();
      extractArgumentsForDynamicLinkParamFile(args, commandlineArgs, paramFileArgs);
      return Pair.of(commandlineArgs, paramFileArgs);
    }
  }

  public CommandLine getCommandLineForStarlark() {
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() throws CommandLineExpansionException {
        return arguments(/* artifactExpander= */ null);
      }

      @Override
      public Iterable<String> arguments(ArtifactExpander artifactExpander)
          throws CommandLineExpansionException {
        if (paramFile == null) {
          return ImmutableList.copyOf(getRawLinkArgv(artifactExpander));
        } else {
          return ImmutableList.<String>builder()
              .add(getLinkerPathString())
              .addAll(splitCommandline(artifactExpander).getSecond())
              .build();
        }
      }
    };
  }

  /**
   * A {@link CommandLine} implementation that returns the command line args pertaining to the
   * .params file.
   */
  @AutoCodec
  @VisibleForSerialization
  static class ParamFileCommandLine extends CommandLine {
    private final Artifact paramsFile;
    private final LinkTargetType linkTargetType;
    private final String forcedToolPath;
    private final FeatureConfiguration featureConfiguration;
    private final String actionName;
    private final CcToolchainVariables variables;

    public ParamFileCommandLine(
        Artifact paramsFile,
        LinkTargetType linkTargetType,
        String forcedToolPath,
        FeatureConfiguration featureConfiguration,
        String actionName,
        CcToolchainVariables variables) {
      this.paramsFile = paramsFile;
      this.linkTargetType = linkTargetType;
      this.forcedToolPath = forcedToolPath;
      this.featureConfiguration = featureConfiguration;
      this.actionName = actionName;
      this.variables = variables;
    }

    @Override
    public Iterable<String> arguments() throws CommandLineExpansionException {
      List<String> argv =
          getRawLinkArgv(
              null, forcedToolPath, featureConfiguration, actionName, linkTargetType, variables);
      return splitCommandline(paramsFile, argv, linkTargetType).getSecond();
    }

    @Override
    public Iterable<String> arguments(ArtifactExpander expander)
        throws CommandLineExpansionException {
      List<String> argv =
          getRawLinkArgv(
              expander,
              forcedToolPath,
              featureConfiguration,
              actionName,
              linkTargetType,
              variables);
      return splitCommandline(paramsFile, argv, linkTargetType).getSecond();
    }
  }

  /** Returns just the .params file portion of the command-line as a {@link CommandLine}. */
  CommandLine paramCmdLine() {
    Preconditions.checkNotNull(paramFile);
    return new ParamFileCommandLine(
        paramFile, linkTargetType, forcedToolPath, featureConfiguration, actionName, variables);
  }

  public static void extractArgumentsForStaticLinkParamFile(
      List<String> args, List<String> commandlineArgs, List<String> paramFileArgs) {
    commandlineArgs.add(args.get(0)); // ar command, must not be moved!
    int argsSize = args.size();
    for (int i = 1; i < argsSize; i++) {
      String arg = args.get(i);
      if (arg.startsWith("@")) {
        commandlineArgs.add(arg); // params file, keep it in the command line
      } else {
        paramFileArgs.add(arg); // the rest goes to the params file
      }
    }
  }

  public static void extractArgumentsForDynamicLinkParamFile(
      List<String> args, List<String> commandlineArgs, List<String> paramFileArgs) {
    // Note, that it is not important that all linker arguments are extracted so that
    // they can be moved into a parameter file, but the vast majority should.
    commandlineArgs.add(args.get(0)); // gcc command, must not be moved!
    int argsSize = args.size();
    for (int i = 1; i < argsSize; i++) {
      String arg = args.get(i);
      if (arg.startsWith("@")) {
        commandlineArgs.add(arg); // params file, keep it in the command line
      } else {
        paramFileArgs.add(arg); // the rest goes to the params file
      }
    }
  }

  /**
   * Returns a raw link command for the given link invocation, including both command and arguments
   * (argv). The version that uses the expander is preferred, but that one can't be used during
   * analysis.
   *
   * @return raw link command line.
   */
  public List<String> getRawLinkArgv() throws CommandLineExpansionException {
    return getRawLinkArgv(null);
  }

  /**
   * Returns a raw link command for the given link invocation, including both command and arguments
   * (argv).
   *
   * @param expander ArtifactExpander for expanding TreeArtifacts.
   * @return raw link command line.
   */
  public List<String> getRawLinkArgv(@Nullable ArtifactExpander expander)
      throws CommandLineExpansionException {
    return getRawLinkArgv(
        expander, forcedToolPath, featureConfiguration, actionName, linkTargetType, variables);
  }

  private static List<String> getRawLinkArgv(
      @Nullable ArtifactExpander expander,
      String forcedToolPath,
      FeatureConfiguration featureConfiguration,
      String actionName,
      LinkTargetType linkTargetType,
      CcToolchainVariables variables)
      throws CommandLineExpansionException {
    List<String> argv = new ArrayList<>();
    if (forcedToolPath != null) {
      argv.add(forcedToolPath);
    } else {
      Preconditions.checkArgument(
          featureConfiguration.actionIsConfigured(actionName),
          String.format("Expected action_config for '%s' to be configured", actionName));
      argv.add(featureConfiguration.getToolPathForAction(linkTargetType.getActionName()));
    }
    try {
      argv.addAll(featureConfiguration.getCommandLine(actionName, variables, expander));
    } catch (ExpansionException e) {
      throw new CommandLineExpansionException(e.getMessage());
    }
    return argv;
  }

  List<String> getCommandLine(@Nullable ArtifactExpander expander)
      throws CommandLineExpansionException {
    // Try to shorten the command line by use of a parameter file.
    // This makes the output with --subcommands (et al) more readable.
    if (paramFile != null) {
      Pair<List<String>, List<String>> split = splitCommandline(expander);
      return split.first;
    } else {
      return getRawLinkArgv(expander);
    }
  }

  @Override
  public List<String> arguments() throws CommandLineExpansionException {
    return getRawLinkArgv(null);
  }

  @Override
  public Iterable<String> arguments(ArtifactExpander artifactExpander)
      throws CommandLineExpansionException {
    return getRawLinkArgv(artifactExpander);
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
    public Builder forceToolPath(String forcedToolPath) {
      this.forcedToolPath = forcedToolPath;
      return this;
    }

    /** Sets the feature configuration for this link action. */
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
    public Builder setLinkerInputArtifacts(NestedSet<Artifact> linkerInputArtifacts) {
      this.linkerInputArtifacts = linkerInputArtifacts;
      return this;
    }

    /**
     * Sets how static the link is supposed to be. For static target types (see {@link
     * LinkTargetType#linkerOrArchiver()}}), the {@link #build} method throws an exception if this
     * is not {@link LinkingMode#STATIC}. The default setting is {@link LinkingMode#STATIC}.
     */
    public Builder setLinkingMode(Link.LinkingMode linkingMode) {
      this.linkingMode = linkingMode;
      return this;
    }

    /**
     * The build info header artifacts are generated header files that are used for link stamping.
     * The {@link #build} method throws an exception if the build info header artifacts are
     * non-empty for a static link (see {@link LinkTargetType#linkerOrArchiver()}}).
     */
    public Builder setBuildInfoHeaderArtifacts(ImmutableList<Artifact> buildInfoHeaderArtifacts) {
      this.buildInfoHeaderArtifacts = buildInfoHeaderArtifacts;
      return this;
    }

    /**
     * Whether the resulting library is intended to be used as a native library from another
     * programming language. This influences the rpath. The {@link #build} method throws an
     * exception if this is true for a static link (see {@link LinkTargetType#linkerOrArchiver()}}).
     */
    public Builder setNativeDeps(boolean nativeDeps) {
      this.nativeDeps = nativeDeps;
      return this;
    }

    /**
     * Sets whether to use test-specific linker flags, e.g. {@code $EXEC_ORIGIN} instead of {@code
     * $ORIGIN} in the rpath or lazy binding.
     */
    public Builder setUseTestOnlyFlags(boolean useTestOnlyFlags) {
      this.useTestOnlyFlags = useTestOnlyFlags;
      return this;
    }

    public Builder setParamFile(Artifact paramFile) {
      this.paramFile = paramFile;
      return this;
    }

    public Builder setBuildVariables(CcToolchainVariables variables) {
      this.variables = variables;
      return this;
    }

    public Builder setToolchainLibrariesSolibDir(PathFragment toolchainLibrariesSolibDir) {
      this.toolchainLibrariesSolibDir = toolchainLibrariesSolibDir;
      return this;
    }

    public Builder setActionName(String actionName) {
      this.actionName = actionName;
      return this;
    }
  }
}
