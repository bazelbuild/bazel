// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import java.util.List;

import javax.annotation.Nullable;

/**
 * Provides information on how to run a linker step. This is used by Link to
 * determine the proper flags to the linker program.
 */
@Immutable
public final class LinkConfigurationImpl implements LinkConfiguration {
  private final BuildConfiguration configuration;
  private final ActionOwner owner;
  private final Artifact output;
  @Nullable private final Artifact interfaceOutput;
  @Nullable private final Artifact symbolCountsOutput;
  private final ImmutableList<Artifact> buildInfoHeaderArtifacts;
  private final Iterable<? extends LinkerInput> linkerInputs;
  private final Iterable<? extends LinkerInput> runtimeInputs;
  private final LinkTargetType linkTargetType;
  private final LinkStaticness linkStaticness;
  private final ImmutableList<String> linkopts;
  private final ImmutableSet<String> features;
  private final ImmutableMap<Artifact, Artifact> linkstamps;
  @Nullable private final PathFragment runtimeSolibDir;
  private final boolean nativeDeps;
  private final boolean useExecOrigin;
  @Nullable private final Artifact interfaceSoBuilder;

  private LinkConfigurationImpl(
      BuildConfiguration configuration,
      ActionOwner owner,
      Artifact output,
      @Nullable Artifact interfaceOutput,
      @Nullable Artifact symbolCountsOutput,
      ImmutableList<Artifact> buildInfoHeaderArtifacts,
      Iterable<? extends LinkerInput> linkerInputs,
      Iterable<? extends LinkerInput> runtimeInputs,
      LinkTargetType linkTargetType,
      LinkStaticness linkStaticness,
      ImmutableList<String> linkopts,
      ImmutableSet<String> features,
      ImmutableMap<Artifact, Artifact> linkstamps,
      @Nullable PathFragment runtimeSolibDir,
      boolean nativeDeps,
      boolean useExecOrigin,
      Artifact interfaceSoBuilder) {
    Preconditions.checkArgument(linkTargetType != LinkTargetType.INTERFACE_DYNAMIC_LIBRARY,
        "you can't link an interface dynamic library directly");
    if (linkTargetType != LinkTargetType.DYNAMIC_LIBRARY) {
      Preconditions.checkArgument(interfaceOutput == null,
          "interface output may only be non-null for dynamic library links");
    }
    if (linkTargetType.isStaticLibraryLink()) {
      Preconditions.checkArgument(linkstamps.isEmpty(),
          "linkstamps may only be present on dynamic library or executable links");
      Preconditions.checkArgument(linkStaticness == LinkStaticness.FULLY_STATIC,
          "static library link must be static");
      Preconditions.checkArgument(buildInfoHeaderArtifacts.isEmpty(),
          "build info headers may only be present on dynamic library or executable links");
      Preconditions.checkArgument(symbolCountsOutput == null,
          "the symbol counts output must be null for static links");
      Preconditions.checkArgument(runtimeSolibDir == null,
          "the runtime solib directory must be null for static links");
      Preconditions.checkArgument(!useExecOrigin,
          "the exec origin flag must be false for static links");
      Preconditions.checkArgument(!nativeDeps,
          "the native deps flag must be false for static links");
    }

    this.configuration = Preconditions.checkNotNull(configuration);
    this.owner = Preconditions.checkNotNull(owner);
    this.output = Preconditions.checkNotNull(output);
    this.interfaceOutput = interfaceOutput;
    this.symbolCountsOutput = symbolCountsOutput;
    this.buildInfoHeaderArtifacts = Preconditions.checkNotNull(buildInfoHeaderArtifacts);
    this.linkerInputs = Preconditions.checkNotNull(linkerInputs);
    this.runtimeInputs = Preconditions.checkNotNull(runtimeInputs);
    this.linkTargetType = Preconditions.checkNotNull(linkTargetType);
    this.linkStaticness = Preconditions.checkNotNull(linkStaticness);
    // For now, silently ignore linkopts if this is a static library link.
    this.linkopts = linkTargetType.isStaticLibraryLink()
        ? ImmutableList.<String>of()
        : Preconditions.checkNotNull(linkopts);
    this.features = Preconditions.checkNotNull(features);
    this.linkstamps = Preconditions.checkNotNull(linkstamps);
    this.runtimeSolibDir = runtimeSolibDir;
    this.nativeDeps = nativeDeps;
    this.useExecOrigin = useExecOrigin;
    // For now, silently ignore interfaceSoBuilder if we don't build an interface dynamic library.
    this.interfaceSoBuilder =
        ((linkTargetType == LinkTargetType.DYNAMIC_LIBRARY) && (interfaceOutput != null))
        ? Preconditions.checkNotNull(interfaceSoBuilder)
        : null;
  }

  @Override
  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  @Override
  public ActionOwner getOwner() {
    return owner;
  }

  @Override
  public Artifact getOutput() {
    return output;
  }

  @Override
  @Nullable public Artifact getInterfaceOutput() {
    return interfaceOutput;
  }

  @Override
  @Nullable public Artifact getSymbolCountsOutput() {
    return symbolCountsOutput;
  }

  @Override
  public ImmutableList<Artifact> getBuildInfoHeaderArtifacts() {
    return buildInfoHeaderArtifacts;
  }

  @Override
  public Iterable<? extends LinkerInput> getLinkerInputs() {
    return linkerInputs;
  }

  @Override
  public Iterable<? extends LinkerInput> getRuntimeInputs() {
    return runtimeInputs;
  }

  @Override
  public LinkTargetType getLinkTargetType() {
    return linkTargetType;
  }

  @Override
  public LinkStaticness getLinkStaticness() {
    return linkStaticness;
  }

  @Override
  public ImmutableList<String> getLinkopts() {
    return linkopts;
  }

  @Override
  public ImmutableSet<String> getFeatures() {
    return features;
  }

  @Override
  public ImmutableMap<Artifact, Artifact> getLinkstamps() {
    return linkstamps;
  }

  @Override
  @Nullable public PathFragment getRuntimeSolibDir() {
    return runtimeSolibDir;
  }

  @Override
  public boolean isNativeDeps() {
    return nativeDeps;
  }

  @Override
  public boolean useExecOrigin() {
    return useExecOrigin;
  }

  @Override
  @Nullable public Artifact buildInterfaceSo() {
    return interfaceSoBuilder;
  }

  public List<String> getRawLinkArgv() {
    return Link.getRawLinkArgv(this);
  }

  public List<String> getArgv() {
    return finalizeWithLinkstampCommands(getRawLinkArgv());
  }

  public List<String> finalizeWithLinkstampCommands(List<String> rawLinkArgv) {
    return Link.finalizeWithLinkstampCommands(this, rawLinkArgv);
  }

  /**
   * A builder for a {@link LinkConfigurationImpl}.
   */
  public static final class Builder {
    private BuildConfiguration configuration;
    private ActionOwner owner;
    @Nullable private Artifact output;
    @Nullable private Artifact interfaceOutput;
    @Nullable private Artifact symbolCountsOutput;
    private ImmutableList<Artifact> buildInfoHeaderArtifacts = ImmutableList.of();
    private Iterable<? extends LinkerInput> linkerInputs = ImmutableList.of();
    private Iterable<? extends LinkerInput> runtimeInputs = ImmutableList.of();
    @Nullable private LinkTargetType linkTargetType;
    private LinkStaticness linkStaticness = LinkStaticness.FULLY_STATIC;
    private ImmutableList<String> linkopts = ImmutableList.of();
    private ImmutableSet<String> features = ImmutableSet.of();
    private ImmutableMap<Artifact, Artifact> linkstamps = ImmutableMap.of();
    @Nullable private PathFragment runtimeSolibDir;
    private boolean nativeDeps;
    private boolean useExecOrigin;
    @Nullable private Artifact interfaceSoBuilder;

    public Builder(BuildConfiguration configuration, ActionOwner owner) {
      this.configuration = configuration;
      this.owner = owner;
    }

    public Builder(RuleContext ruleContext) {
      this(ruleContext.getConfiguration(), ruleContext.getActionOwner());
    }

    public LinkConfigurationImpl build() {
      return new LinkConfigurationImpl(configuration, owner, output, interfaceOutput,
          symbolCountsOutput, buildInfoHeaderArtifacts, linkerInputs, runtimeInputs, linkTargetType,
          linkStaticness, linkopts, features, linkstamps, runtimeSolibDir, nativeDeps,
          useExecOrigin, interfaceSoBuilder);
    }

    /**
     * Sets the type of the link. It is an error to try to set this to {@link
     * LinkTargetType#INTERFACE_DYNAMIC_LIBRARY}. Note that all the static target types (see {@link
     * LinkTargetType#isStaticLibraryLink}) are equivalent, and there is no check that the output
     * artifact matches the target type extension.
     */
    public Builder setLinkTargetType(LinkTargetType linkTargetType) {
      Preconditions.checkArgument(linkTargetType != LinkTargetType.INTERFACE_DYNAMIC_LIBRARY);
      this.linkTargetType = linkTargetType;
      return this;
    }

    /**
     * Sets the primary output artifact. This must be called before calling {@link #build}.
     */
    public Builder setOutput(Artifact output) {
      this.output = output;
      return this;
    }

    /**
     * Sets a list of linker inputs. These get turned into linker options depending on the
     * staticness and the target type.
     */
    public Builder setLinkerInputs(Iterable<LinkerInput> linkerInputs) {
      this.linkerInputs = CollectionUtils.makeImmutable(linkerInputs);
      return this;
    }

    public Builder setRuntimeInputs(Iterable<LinkerInput> runtimeInputs) {
      this.runtimeInputs = CollectionUtils.makeImmutable(runtimeInputs);
      return this;
    }

    /**
     * Sets the additional interface output artifact, which is only used for dynamic libraries. The
     * {@link #build} method throws an exception if the target type is not {@link
     * LinkTargetType#DYNAMIC_LIBRARY}.
     */
    public Builder setInterfaceOutput(Artifact interfaceOutput) {
      this.interfaceOutput = interfaceOutput;
      return this;
    }

    /**
     * Sets an additional output artifact that contains symbol counts. The {@link #build} method
     * throws an exception if this is non-null for a static link (see
     * {@link LinkTargetType#isStaticLibraryLink}).
     */
    public Builder setSymbolCountsOutput(Artifact symbolCountsOutput) {
      this.symbolCountsOutput = symbolCountsOutput;
      return this;
    }

    /**
     * Sets the linker options. These are passed to the linker in addition to the other linker
     * options like linker inputs, symbol count options, etc. The {@link #build} method
     * throws an exception if the linker options are non-empty for a static link (see {@link
     * LinkTargetType#isStaticLibraryLink}).
     */
    public Builder setLinkopts(ImmutableList<String> linkopts) {
      this.linkopts = linkopts;
      return this;
    }

    /**
     * Sets how static the link is supposed to be. For static target types (see {@link
     * LinkTargetType#isStaticLibraryLink}), the {@link #build} method throws an exception if this
     * is not {@link LinkStaticness#FULLY_STATIC}. The default setting is {@link
     * LinkStaticness#FULLY_STATIC}.
     */
    public Builder setLinkStaticness(LinkStaticness linkStaticness) {
      this.linkStaticness = linkStaticness;
      return this;
    }

    /**
     * Sets the binary that should be used to create the interface output for a dynamic library.
     * This is ignored unless the target type is {@link LinkTargetType#DYNAMIC_LIBRARY} and an
     * interface output artifact is specified.
     */
    public Builder setInterfaceSoBuilder(Artifact interfaceSoBuilder) {
      this.interfaceSoBuilder = interfaceSoBuilder;
      return this;
    }

    /**
     * Sets the linkstamps. Linkstamps are additional C++ source files that are compiled as part of
     * the link command. The {@link #build} method throws an exception if the linkstamps are
     * non-empty for a static link (see {@link LinkTargetType#isStaticLibraryLink}).
     */
    public Builder setLinkstamps(ImmutableMap<Artifact, Artifact> linkstamps) {
      this.linkstamps = linkstamps;
      return this;
    }

    /**
     * The build info header artifacts are generated header files that are used for link stamping.
     * The {@link #build} method throws an exception if the build info header artifacts are
     * non-empty for a static link (see {@link LinkTargetType#isStaticLibraryLink}).
     */
    public Builder setBuildInfoHeaderArtifacts(ImmutableList<Artifact> buildInfoHeaderArtifacts) {
      this.buildInfoHeaderArtifacts = buildInfoHeaderArtifacts;
      return this;
    }

    /**
     * Sets the features enabled for the rule.
     */
    public Builder setFeatures(ImmutableSet<String> features) {
      this.features = features;
      return this;
    }

    /**
     * Sets the directory of the dynamic runtime libraries, which is added to the rpath. The {@link
     * #build} method throws an exception if the runtime dir is non-null for a static link (see
     * {@link LinkTargetType#isStaticLibraryLink}).
     */
    public Builder setRuntimeSolibDir(PathFragment runtimeSolibDir) {
      this.runtimeSolibDir = runtimeSolibDir;
      return this;
    }

    /**
     * Whether the resulting library is intended to be used as a native library from another
     * programming language. This influences the rpath. The {@link #build} method throws an
     * exception if this is true for a static link (see {@link LinkTargetType#isStaticLibraryLink}).
     */
    public Builder setNativeDeps(boolean nativeDeps) {
      this.nativeDeps = nativeDeps;
      return this;
    }

    /**
     * Sets whether to use {@code $EXEC_ORIGIN} instead of {@code $ORIGIN} in the rpath. This
     * requires a dynamic linker that support this feature. The {@link #build} method throws an
     * exception if this is true for a static link (see {@link LinkTargetType#isStaticLibraryLink}).
     */
    public Builder setUseExecOrigin(boolean useExecOrigin) {
      this.useExecOrigin = useExecOrigin;
      return this;
    }
  }
}
