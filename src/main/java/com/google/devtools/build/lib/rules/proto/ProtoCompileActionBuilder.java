// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.Iterables.isEmpty;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.rules.proto.ProtoCommon.areDepsStrict;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.actions.CommandLineItem.CapturingMapFn;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.stringtemplate.ExpansionException;
import com.google.devtools.build.lib.analysis.stringtemplate.TemplateContext;
import com.google.devtools.build.lib.analysis.stringtemplate.TemplateExpander;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** Constructs actions to run the protocol compiler to generate sources from .proto files. */
public class ProtoCompileActionBuilder {
  @VisibleForTesting
  public static final String STRICT_DEPS_FLAG_TEMPLATE =
      "--direct_dependencies_violation_msg=" + StrictProtoDepsViolationMessage.MESSAGE;

  private static final String MNEMONIC = "GenProto";
  private static final Action[] NO_ACTIONS = new Action[0];

  private final RuleContext ruleContext;
  private final ProtoSourcesProvider protoSourcesProvider;
  private final FilesToRunProvider protoCompiler;
  private final String language;
  private final String langPrefix;
  private final Iterable<Artifact> outputs;
  private Iterable<Artifact> inputs;
  private String langParameter;
  private String langPluginName;
  private String langPluginParameter;
  private Supplier<String> langPluginParameterSupplier;
  private boolean hasServices;
  private Iterable<String> additionalCommandLineArguments;
  private Iterable<FilesToRunProvider> additionalTools;
  private boolean checkStrictImportPublic;

  public ProtoCompileActionBuilder allowServices(boolean hasServices) {
    this.hasServices = hasServices;
    return this;
  }

  public ProtoCompileActionBuilder setInputs(Iterable<Artifact> inputs) {
    this.inputs = inputs;
    return this;
  }

  public ProtoCompileActionBuilder setLangParameter(String langParameter) {
    this.langParameter = langParameter;
    return this;
  }

  public ProtoCompileActionBuilder setLangPluginName(String langPluginName) {
    this.langPluginName = langPluginName;
    return this;
  }

  public ProtoCompileActionBuilder setLangPluginParameter(String langPluginParameter) {
    this.langPluginParameter = langPluginParameter;
    return this;
  }

  public ProtoCompileActionBuilder setLangPluginParameterSupplier(
      Supplier<String> langPluginParameterSupplier) {
    this.langPluginParameterSupplier = langPluginParameterSupplier;
    return this;
  }

  public ProtoCompileActionBuilder setAdditionalCommandLineArguments(
      Iterable<String> additionalCmdLine) {
    this.additionalCommandLineArguments = additionalCmdLine;
    return this;
  }

  public ProtoCompileActionBuilder setAdditionalTools(
      Iterable<FilesToRunProvider> additionalTools) {
    this.additionalTools = additionalTools;
    return this;
  }

  public ProtoCompileActionBuilder checkStrictImportPublic(boolean checkStrictImportPublic) {
    this.checkStrictImportPublic = checkStrictImportPublic;
    return this;
  }

  public ProtoCompileActionBuilder(
      RuleContext ruleContext,
      ProtoSourcesProvider protoSourcesProvider,
      FilesToRunProvider protoCompiler,
      String language,
      String langPrefix,
      Iterable<Artifact> outputs) {
    this.ruleContext = ruleContext;
    this.protoSourcesProvider = protoSourcesProvider;
    this.protoCompiler = protoCompiler;
    this.language = language;
    this.langPrefix = langPrefix;
    this.outputs = outputs;
  }

  /** Static class to avoid keeping a reference to this builder after build() is called. */
  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class LazyLangPluginFlag extends LazyString {
    private final String langPrefix;
    private final Supplier<String> langPluginParameter;

    @AutoCodec.VisibleForSerialization
    LazyLangPluginFlag(String langPrefix, Supplier<String> langPluginParameter) {
      this.langPrefix = langPrefix;
      this.langPluginParameter = langPluginParameter;
    }

    @Override
    public String toString() {
      return String.format("--%s_out=%s", langPrefix, langPluginParameter.get());
    }
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class LazyCommandLineExpansion extends LazyString {
    // E.g., --java_out=%s
    private final String template;
    private final Map<String, ? extends CharSequence> variableValues;

    @AutoCodec.VisibleForSerialization
    LazyCommandLineExpansion(String template, Map<String, ? extends CharSequence> variableValues) {
      this.template = template;
      this.variableValues = variableValues;
    }

    @Override
    public String toString() {
      try {
        return TemplateExpander.expand(
            template,
            new TemplateContext() {
              @Override
              public String lookupVariable(String name)
                  throws ExpansionException {
                CharSequence value = variableValues.get(name);
                if (value == null) {
                  throw new ExpansionException(String.format("$(%s) not defined", name));
                }
                return value.toString();
              }

              @Override
              public String lookupFunction(String name, String param) throws ExpansionException {
                throw new ExpansionException(String.format("$(%s) not defined", name));
              }
            });
      } catch (ExpansionException e) {
        // Squeelch. We don't throw this exception in the lookupMakeVariable implementation above,
        // and we can't report it here anyway, because this code will typically execute in the
        // Execution phase.
      }
      return template;
    }
  }

  public Action[] build() {
    checkState(
        langPluginParameter == null || langPluginParameterSupplier == null,
        "Only one of {langPluginParameter, langPluginParameterSupplier} should be set.");

    if (isEmpty(outputs)) {
      return NO_ACTIONS;
    }

    try {
      return createAction().build(ruleContext);
    } catch (MissingPrerequisiteException e) {
      return NO_ACTIONS;
    }
  }

  private SpawnAction.Builder createAction() throws MissingPrerequisiteException {
    SpawnAction.Builder result =
        new SpawnAction.Builder()
            .addTransitiveInputs(protoSourcesProvider.getTransitiveProtoSources());

    FilesToRunProvider langPluginTarget = getLangPluginTarget();
    if (langPluginTarget != null) {
      result.addTool(langPluginTarget);
    }

    if (inputs != null) {
      result.addInputs(inputs);
    }

    if (this.additionalTools != null) {
      for (FilesToRunProvider tool : additionalTools) {
        result.addTool(tool);
      }
    }

    if (protoCompiler == null) {
      throw new MissingPrerequisiteException();
    }

    result
        .addOutputs(outputs)
        .setResources(AbstractAction.DEFAULT_RESOURCE_SET)
        .useDefaultShellEnvironment()
        .setExecutable(protoCompiler)
        .addCommandLine(
            createProtoCompilerCommandLine().build(),
            ParamFileInfo.builder(ParameterFileType.UNQUOTED).build())
        .setProgressMessage("Generating %s proto_library %s", language, ruleContext.getLabel())
        .setMnemonic(MNEMONIC);

    return result;
  }

  @Nullable
  private FilesToRunProvider getLangPluginTarget() throws MissingPrerequisiteException {
    if (langPluginName == null) {
      return null;
    }
    FilesToRunProvider result =
        ruleContext.getExecutablePrerequisite(langPluginName, RuleConfiguredTarget.Mode.HOST);
    if (result == null) {
      throw new MissingPrerequisiteException();
    }
    return result;
  }

  /** Commandline generator for protoc invocations. */
  @VisibleForTesting
  CustomCommandLine.Builder createProtoCompilerCommandLine() throws MissingPrerequisiteException {
    CustomCommandLine.Builder result = CustomCommandLine.builder();

    if (langPluginName == null) {
      if (langParameter != null) {
        result.addDynamicString(langParameter);
      }
    } else {
      FilesToRunProvider langPluginTarget = getLangPluginTarget();
      Supplier<String> langPluginParameter1 =
          langPluginParameter == null
              ? langPluginParameterSupplier
              : Suppliers.ofInstance(langPluginParameter);

      Preconditions.checkArgument(langParameter == null);
      Preconditions.checkArgument(langPluginParameter1 != null);
      // We pass a separate langPluginName as there are plugins that cannot be overridden
      // and thus we have to deal with "$xx_plugin" and "xx_plugin".
      result.addFormatted(
          "--plugin=protoc-gen-%s=%s", langPrefix, langPluginTarget.getExecutable().getExecPath());
      result.addLazyString(new LazyLangPluginFlag(langPrefix, langPluginParameter1));
    }

    result.addAll(ruleContext.getFragment(ProtoConfiguration.class).protocOpts());

    boolean areDepsStrict = areDepsStrict(ruleContext);

    // Add include maps
    addIncludeMapArguments(
        result,
        areDepsStrict ? protoSourcesProvider.getProtosInDirectDeps() : null,
        protoSourcesProvider.getDirectProtoSourceRoots(),
        protoSourcesProvider.getTransitiveProtoSources());

    if (areDepsStrict) {
      // Note: the %s in the line below is used by proto-compiler. That is, the string we create
      // here should have a literal %s in it.
      result.addFormatted(STRICT_DEPS_FLAG_TEMPLATE, ruleContext.getLabel());
    }

    for (Artifact src : protoSourcesProvider.getDirectProtoSources()) {
      result.addPath(src.getRootRelativePath());
    }

    if (!hasServices) {
      result.add("--disallow_services");
    }
    if (checkStrictImportPublic) {
      NestedSet<Artifact> protosInExports = protoSourcesProvider.getProtosInExports();
      if (protosInExports.isEmpty()) {
        // This line is necessary to trigger the check.
        result.add("--allowed_public_imports=");
      } else {
        result.addAll(
            "--allowed_public_imports",
            VectorArg.join(":")
                .each(protosInExports)
                .mapped(new ExpandToPathFn(protoSourcesProvider.getTransitiveProtoSourceRoots())));
      }
    }

    if (additionalCommandLineArguments != null) {
      result.addAll(ImmutableList.copyOf(additionalCommandLineArguments));
    }

    return result;
  }

  /** Signifies that a prerequisite could not be satisfied. */
  private static class MissingPrerequisiteException extends Exception {}

  public static void writeDescriptorSet(
      RuleContext ruleContext,
      final CharSequence outReplacement,
      Collection<Artifact> protosToCompile,
      NestedSet<Artifact> transitiveSources,
      NestedSet<Artifact> protosInDirectDeps,
      Artifact output,
      boolean allowServices,
      NestedSet<Artifact> transitiveDescriptorSets,
      NestedSet<String> protoSourceRoots,
      NestedSet<String> directProtoSourceRoots) {
    if (protosToCompile.isEmpty()) {
      ruleContext.registerAction(
          FileWriteAction.createEmptyWithInputs(
              ruleContext.getActionOwner(), transitiveDescriptorSets, output));
      return;
    }

    SpawnAction.Builder actions =
        createActions(
            ruleContext,
            ImmutableList.of(createDescriptorSetToolchain(outReplacement)),
            protosToCompile,
            transitiveSources,
            protosInDirectDeps,
            /* protosInExports= */ null,
            protoSourceRoots,
            directProtoSourceRoots,
            /* exportedProtoSourceRoots= */ null,
            ruleContext.getLabel(),
            ImmutableList.of(output),
            "Descriptor Set",
            allowServices);
    if (actions == null) {
      return;
    }

    actions.setMnemonic("GenProtoDescriptorSet");
    actions.addTransitiveInputs(transitiveDescriptorSets);
    ruleContext.registerAction(actions.build(ruleContext));
  }

  private static ToolchainInvocation createDescriptorSetToolchain(CharSequence outReplacement) {
    return new ToolchainInvocation(
        "dontcare",
        ProtoLangToolchainProvider.create(
            // Note: adding --include_imports here was requested multiple times, but it'll cause the
            // output size to become quadratic, so don't.
            // A rule that concatenates the artifacts from ctx.deps.proto.transitive_descriptor_sets
            // provides similar results.
            "--descriptor_set_out=$(OUT)",
            /* pluginExecutable= */ null,
            /* runtime= */ null,
            /* blacklistedProtos= */ NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER)),
        outReplacement);
  }

  public static void registerActions(
      RuleContext ruleContext,
      List<ToolchainInvocation> toolchainInvocations,
      Iterable<Artifact> protosToCompile,
      NestedSet<Artifact> transitiveSources,
      NestedSet<Artifact> protosInDirectDeps,
      NestedSet<String> protoSourceRoots,
      NestedSet<String> directProtoSourceRoots,
      Label ruleLabel,
      Iterable<Artifact> outputs,
      String flavorName,
      boolean allowServices) {
    registerActions(
        ruleContext,
        toolchainInvocations,
        protosToCompile,
        transitiveSources,
        protosInDirectDeps,
        protoSourceRoots,
        directProtoSourceRoots,
        ruleLabel,
        outputs,
        flavorName,
        allowServices,
        /* protosInExports= */ null,
        /* exportedProtoSourceRoots= */ null);
  }

  /**
   * Registers actions to generate code from .proto files.
   *
   * <p>This method uses information from proto_lang_toolchain() rules. New rules should use this
   * method instead of the soup of methods above.
   *
   * @param toolchainInvocations See {@link #createCommandLineFromToolchains}.
   * @param ruleLabel See {@link #createCommandLineFromToolchains}.
   * @param outputs The artifacts that the resulting action must create.
   * @param flavorName e.g., "Java (Immutable)"
   * @param allowServices If false, the compilation will break if any .proto file has service
   */
  public static void registerActions(
      RuleContext ruleContext,
      List<ToolchainInvocation> toolchainInvocations,
      Iterable<Artifact> protosToCompile,
      NestedSet<Artifact> transitiveSources,
      NestedSet<Artifact> protosInDirectDeps,
      NestedSet<String> protoSourceRoots,
      NestedSet<String> directProtoSourceRoots,
      Label ruleLabel,
      Iterable<Artifact> outputs,
      String flavorName,
      boolean allowServices,
      NestedSet<Artifact> protosInExports,
      NestedSet<String> exportedProtoSourceRoots) {
    SpawnAction.Builder actions =
        createActions(
            ruleContext,
            toolchainInvocations,
            protosToCompile,
            transitiveSources,
            protosInDirectDeps,
            protosInExports,
            protoSourceRoots,
            directProtoSourceRoots,
            exportedProtoSourceRoots,
            ruleLabel,
            outputs,
            flavorName,
            allowServices);
    if (actions != null) {
      ruleContext.registerAction(actions.build(ruleContext));
    }
  }

  @Nullable
  private static SpawnAction.Builder createActions(
      RuleContext ruleContext,
      List<ToolchainInvocation> toolchainInvocations,
      Iterable<Artifact> protosToCompile,
      NestedSet<Artifact> transitiveSources,
      @Nullable NestedSet<Artifact> protosInDirectDeps,
      @Nullable NestedSet<Artifact> protosInExports,
      NestedSet<String> protoSourceRoots,
      NestedSet<String> directProtoSourceRoots,
      @Nullable NestedSet<String> exportedProtoSourceRoots,
      Label ruleLabel,
      Iterable<Artifact> outputs,
      String flavorName,
      boolean allowServices) {

    if (isEmpty(outputs)) {
      return null;
    }

    SpawnAction.Builder result = new SpawnAction.Builder().addTransitiveInputs(transitiveSources);

    for (ToolchainInvocation invocation : toolchainInvocations) {
      ProtoLangToolchainProvider toolchain = invocation.toolchain;
      if (toolchain.pluginExecutable() != null) {
        result.addTool(toolchain.pluginExecutable());
      }
    }

    FilesToRunProvider compilerTarget =
        ruleContext.getExecutablePrerequisite(":proto_compiler", RuleConfiguredTarget.Mode.HOST);
    if (compilerTarget == null) {
      return null;
    }

    result
        .addOutputs(outputs)
        .setResources(AbstractAction.DEFAULT_RESOURCE_SET)
        .useDefaultShellEnvironment()
        .setExecutable(compilerTarget)
        .addCommandLine(
            createCommandLineFromToolchains(
                toolchainInvocations,
                protosToCompile,
                transitiveSources,
                protoSourceRoots,
                directProtoSourceRoots,
                exportedProtoSourceRoots,
                areDepsStrict(ruleContext) ? protosInDirectDeps : null,
                arePublicImportsStrict(ruleContext) ? protosInExports : null,
                ruleLabel,
                allowServices,
                ruleContext.getFragment(ProtoConfiguration.class).protocOpts()),
            ParamFileInfo.builder(ParameterFileType.UNQUOTED).build())
        .setProgressMessage("Generating %s proto_library %s", flavorName, ruleContext.getLabel())
        .setMnemonic(MNEMONIC);

    return result;
  }

  public static boolean arePublicImportsStrict(RuleContext ruleContext) {
    return ruleContext.getFragment(ProtoConfiguration.class).strictPublicImports();
  }

  /**
   * Constructs command-line arguments to execute proto-compiler.
   *
   * <ul>
   *   <li>Each toolchain contributes a command-line, formatted from its commandLine() method.
   *   <li>$(OUT) is replaced with the outReplacement field of ToolchainInvocation.
   *   <li>$(PLUGIN_out) is replaced with PLUGIN_<key>_out where 'key' is the key of
   *       toolchainInvocations. The key thus allows multiple plugins in one command-line.
   *   <li>If a toolchain's {@code plugin()} is non-null, we point at it by emitting
   *       --plugin=protoc-gen-PLUGIN_<key>=<location of plugin>.
   * </ul>
   *
   * Note {@code toolchainInvocations} is ordered, and affects the order in which plugins are
   * called. As some plugins rely on output from other plugins, their order matters.
   *
   * @param toolchainInvocations See {@link #createCommandLineFromToolchains}.
   * @param ruleLabel Name of the proto_library for which we're compiling. This string is used to
   *     populate an error message format that's passed to proto-compiler.
   * @param allowServices If false, the compilation will break if any .proto file has
   */
  @VisibleForTesting
  static CustomCommandLine createCommandLineFromToolchains(
      List<ToolchainInvocation> toolchainInvocations,
      Iterable<Artifact> protosToCompile,
      NestedSet<Artifact> transitiveSources,
      NestedSet<String> transitiveProtoPathFlags,
      NestedSet<String> directProtoSourceRoots,
      NestedSet<String> exportedProtoSourceRoots,
      @Nullable NestedSet<Artifact> protosInDirectDeps,
      @Nullable NestedSet<Artifact> protosInExports,
      Label ruleLabel,
      boolean allowServices,
      ImmutableList<String> protocOpts) {
    CustomCommandLine.Builder cmdLine = CustomCommandLine.builder();

    cmdLine.addAll(
        VectorArg.of(transitiveProtoPathFlags).mapped(EXPAND_TRANSITIVE_PROTO_PATH_FLAGS));

    // A set to check if there are multiple invocations with the same name.
    HashSet<String> invocationNames = new HashSet<>();

    for (ToolchainInvocation invocation : toolchainInvocations) {
      if (!invocationNames.add(invocation.name)) {
        throw new IllegalStateException(
            "Invocation name "
                + invocation.name
                + " appears more than once. "
                + "This could lead to incorrect proto-compiler behavior");
      }

      ProtoLangToolchainProvider toolchain = invocation.toolchain;

      cmdLine.addLazyString(
          new LazyCommandLineExpansion(
              toolchain.commandLine(),
              ImmutableMap.of(
                  "OUT",
                  invocation.outReplacement,
                  "PLUGIN_OUT",
                  String.format("PLUGIN_%s_out", invocation.name))));

      if (toolchain.pluginExecutable() != null) {
        cmdLine.addFormatted(
            "--plugin=protoc-gen-PLUGIN_%s=%s",
            invocation.name, toolchain.pluginExecutable().getExecutable().getExecPath());
      }
    }

    cmdLine.addAll(protocOpts);

    // Add include maps
    addIncludeMapArguments(cmdLine, protosInDirectDeps, directProtoSourceRoots, transitiveSources);

    if (protosInDirectDeps != null) {
      cmdLine.addFormatted(STRICT_DEPS_FLAG_TEMPLATE, ruleLabel);
    }

    if (protosInExports != null) {
      if (protosInExports.isEmpty()) {
        // This line is necessary to trigger the check.
        cmdLine.add("--allowed_public_imports=");
      } else {
        cmdLine.addAll(
            "--allowed_public_imports",
            VectorArg.join(":")
                .each(protosInExports)
                .mapped(new ExpandToPathFn(exportedProtoSourceRoots)));
      }
    }

    for (Artifact src : protosToCompile) {
      cmdLine.addPath(src.getExecPath());
    }

    if (!allowServices) {
      cmdLine.add("--disallow_services");
    }

    return cmdLine.build();
  }

  @VisibleForTesting
  static void addIncludeMapArguments(
      CustomCommandLine.Builder commandLine,
      @Nullable NestedSet<Artifact> protosInDirectDependencies,
      NestedSet<String> directProtoSourceRoots,
      NestedSet<Artifact> transitiveImports) {
    // For each import, include both the import as well as the import relativized against its
    // protoSourceRoot. This ensures that protos can reference either the full path or the short
    // path when including other protos.
    commandLine.addAll(
        VectorArg.of(transitiveImports).mapped(new ExpandImportArgsFn(directProtoSourceRoots)));
    if (protosInDirectDependencies != null) {
      if (!protosInDirectDependencies.isEmpty()) {
        commandLine.addAll(
            "--direct_dependencies",
            VectorArg.join(":")
                .each(protosInDirectDependencies)
                .mapped(new ExpandToPathFn(directProtoSourceRoots)));

      } else {
        // The proto compiler requires an empty list to turn on strict deps checking
        commandLine.add("--direct_dependencies=");
      }
    }
  }

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final CommandLineItem.MapFn<String> EXPAND_TRANSITIVE_PROTO_PATH_FLAGS =
      (flag, args) -> args.accept("--proto_path=" + flag);

  @AutoCodec
  @AutoCodec.VisibleForSerialization
  static final class ExpandImportArgsFn implements CapturingMapFn<Artifact> {
    private final NestedSet<String> directProtoSourceRoots;

    public ExpandImportArgsFn(NestedSet<String> directProtoSourceRoots) {
      this.directProtoSourceRoots = directProtoSourceRoots;
    }

    /**
     * Generates up to two import flags for each artifact: one for full path (only relative to the
     * repository root) and one for the path relative to the proto source root (if one exists
     * corresponding to the artifact).
     */
    @Override
    public void expandToCommandLine(Artifact proto, Consumer<String> args) {
      for (String directProtoSourceRoot : directProtoSourceRoots) {
        String path = getPathIgnoringSourceRoot(proto, directProtoSourceRoot);
        if (path != null) {
          args.accept("-I" + path + "=" + proto.getExecPathString());
        }
      }
      args.accept("-I" + getPathIgnoringRepository(proto) + "=" + proto.getExecPathString());
    }
  }

  @AutoCodec
  @AutoCodec.VisibleForSerialization
  static final class ExpandToPathFn implements CapturingMapFn<Artifact> {
    private final NestedSet<String> directProtoSourceRoots;

    public ExpandToPathFn(NestedSet<String> directProtoSourceRoots) {
      this.directProtoSourceRoots = directProtoSourceRoots;
    }

    @Override
    public void expandToCommandLine(Artifact proto, Consumer<String> args) {
      for (String directProtoSourceRoot : directProtoSourceRoots) {
        String path = getPathIgnoringSourceRoot(proto, directProtoSourceRoot);
        if (path != null) {
          args.accept(path);
        }
      }
      args.accept(getPathIgnoringRepository(proto));
    }
  }

  /**
   * Gets the artifact's path relative to the root, ignoring the external repository the artifact is
   * at. For example, <code>
   * //a:b.proto --> a/b.proto
   * {@literal @}foo//a:b.proto --> a/b.proto
   * </code>
   */
  private static String getPathIgnoringRepository(Artifact artifact) {
    return artifact
        .getRootRelativePath()
        .relativeTo(
            artifact.getOwnerLabel().getPackageIdentifier().getRepository().getPathUnderExecRoot())
        .toString();
  }

  /**
   * Gets the artifact's path relative to the proto source root, ignoring the external repository
   * the artifact is at. For example, <code>
   * //a/b/c:d.proto with proto source root a/b --> c/d.proto
   * {@literal @}foo//a/b/c:d.proto with proto source root a/b --> c/d.proto
   * </code>
   */
  private static String getPathIgnoringSourceRoot(Artifact artifact, String directProtoSourceRoot) {
    // TODO(bazel-team): IAE is caught here because every artifact is relativized against every
    // directProtoSourceRoot. Instead of catching the exception, a check should be performed
    // to see if the artifact has the root as a substring before relativizing.
    try {
      return PathFragment.createAlreadyNormalized(getPathIgnoringRepository(artifact))
          .relativeTo(directProtoSourceRoot)
          .toString();
    } catch (IllegalArgumentException exception) {
      // do nothing
    }
    return null;
  }

  /**
   * Describes a toolchain and the value to replace for a $(OUT) that might appear in its
   * commandLine() (e.g., "bazel-out/foo.srcjar").
   */
  public static class ToolchainInvocation {
    final String name;
    public final ProtoLangToolchainProvider toolchain;
    final CharSequence outReplacement;

    public ToolchainInvocation(
        String name, ProtoLangToolchainProvider toolchain, CharSequence outReplacement) {
      checkState(!name.contains(" "), "Name %s should not contain spaces", name);
      this.name = name;
      this.toolchain = toolchain;
      this.outReplacement = outReplacement;
    }
  }
}
