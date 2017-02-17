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
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.MakeVariableExpander;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.LazyString;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Constructs actions to run the protocol compiler to generate sources from .proto files. */
public class ProtoCompileActionBuilder {
  private static final String MNEMONIC = "GenProto";
  private static final ResourceSet GENPROTO_RESOURCE_SET =
      ResourceSet.createWithRamCpuIo(100, .1, .0);
  private static final Action[] NO_ACTIONS = new Action[0];

  private RuleContext ruleContext;
  private SupportData supportData;
  private String language;
  private String langPrefix;
  private Iterable<Artifact> outputs;
  private String langParameter;
  private String langPluginName;
  private String langPluginParameter;
  private Supplier<String> langPluginParameterSupplier;
  private boolean hasServices;
  private Iterable<String> additionalCommandLineArguments;
  private Iterable<FilesToRunProvider> additionalTools;

  /** Build a proto compiler commandline argument for use in setXParameter methods. */
  public static String buildProtoArg(String arg, String value, Iterable<String> flags) {
    return String.format(
        "--%s=%s%s", arg, (isEmpty(flags) ? "" : Joiner.on(',').join(flags) + ":"), value);
  }

  public ProtoCompileActionBuilder setRuleContext(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    return this;
  }

  public ProtoCompileActionBuilder setSupportData(SupportData supportData) {
    this.supportData = supportData;
    return this;
  }

  public ProtoCompileActionBuilder setLanguage(String language) {
    this.language = language;
    return this;
  }

  public ProtoCompileActionBuilder setLangPrefix(String langPrefix) {
    this.langPrefix = langPrefix;
    return this;
  }

  public ProtoCompileActionBuilder allowServices(boolean hasServices) {
    this.hasServices = hasServices;
    return this;
  }

  public ProtoCompileActionBuilder setOutputs(Iterable<Artifact> outputs) {
    this.outputs = outputs;
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

  public ProtoCompileActionBuilder(
      RuleContext ruleContext,
      SupportData supportData,
      String language,
      String langPrefix,
      Iterable<Artifact> outputs) {
    this.ruleContext = ruleContext;
    this.supportData = supportData;
    this.language = language;
    this.langPrefix = langPrefix;
    this.outputs = outputs;
  }

  /** Static class to avoid keeping a reference to this builder after build() is called. */
  private static class LazyLangPluginFlag extends LazyString {
    private final String langPrefix;
    private final Supplier<String> langPluginParameter1;

    LazyLangPluginFlag(String langPrefix, Supplier<String> langPluginParameter1) {
      this.langPrefix = langPrefix;
      this.langPluginParameter1 = langPluginParameter1;
    }

    @Override
    public String toString() {
      return String.format("--%s_out=%s", langPrefix, langPluginParameter1.get());
    }
  }

  private static class LazyCommandLineExpansion extends LazyString {
    // E.g., --java_out=%s
    private final String template;
    private final Map<String, ? extends CharSequence> variableValues;

    private LazyCommandLineExpansion(
        String template, Map<String, ? extends CharSequence> variableValues) {
      this.template = template;
      this.variableValues = variableValues;
    }

    @Override
    public String toString() {
      try {
        return MakeVariableExpander.expand(
            template,
            new MakeVariableExpander.Context() {
              @Override
              public String lookupMakeVariable(String var)
                  throws MakeVariableExpander.ExpansionException {
                CharSequence value = variableValues.get(var);
                return value != null ? value.toString() : var;
              }
            });
      } catch (MakeVariableExpander.ExpansionException e) {
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
        new SpawnAction.Builder().addTransitiveInputs(supportData.getTransitiveImports());

    FilesToRunProvider langPluginTarget = getLangPluginTarget();
    if (langPluginTarget != null) {
      result.addTool(langPluginTarget);
    }

    FilesToRunProvider compilerTarget =
        ruleContext.getExecutablePrerequisite(":proto_compiler", RuleConfiguredTarget.Mode.HOST);
    if (compilerTarget == null) {
      throw new MissingPrerequisiteException();
    }

    if (this.additionalTools != null) {
      for (FilesToRunProvider tool : additionalTools) {
        result.addTool(tool);
      }
    }

    result
        .useParameterFile(ParameterFile.ParameterFileType.UNQUOTED)
        .addOutputs(outputs)
        .setResources(GENPROTO_RESOURCE_SET)
        .useDefaultShellEnvironment()
        .setExecutable(compilerTarget)
        .setCommandLine(createProtoCompilerCommandLine().build())
        .setProgressMessage("Generating " + language + " proto_library " + ruleContext.getLabel())
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
        result.add(langParameter);
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
      result.add(
          String.format(
              "--plugin=protoc-gen-%s=%s",
              langPrefix, langPluginTarget.getExecutable().getExecPathString()));
      result.add(new LazyLangPluginFlag(langPrefix, langPluginParameter1));
    }

    result.add(ruleContext.getFragment(ProtoConfiguration.class).protocOpts());

    boolean areDepsStrict = areDepsStrict(ruleContext);

    // Add include maps
    result.add(
        new ProtoCommandLineArgv(
            areDepsStrict ? supportData.getProtosInDirectDeps() : null,
            supportData.getTransitiveImports()));

    if (areDepsStrict) {
      // Note: the %s in the line below is used by proto-compiler. That is, the string we create
      // here should have a literal %s in it.
      result.add(
          createStrictProtoDepsViolationErrorMessage(ruleContext.getLabel().getCanonicalForm()));
    }

    for (Artifact src : supportData.getDirectProtoSources()) {
      result.addPath(src.getRootRelativePath());
    }

    if (!hasServices) {
      result.add("--disallow_services");
    }

    if (additionalCommandLineArguments != null) {
      result.add(additionalCommandLineArguments);
    }

    return result;
  }

  /**
   * Static inner class since these objects live into the execution phase and so they must not keep
   * alive references to the surrounding analysis-phase objects.
   */
  @VisibleForTesting
  static class ProtoCommandLineArgv extends CustomCommandLine.CustomMultiArgv {
    @Nullable private final Iterable<Artifact> protosInDirectDependencies;
    private final Iterable<Artifact> transitiveImports;

    ProtoCommandLineArgv(
        @Nullable Iterable<Artifact> protosInDirectDependencies,
        Iterable<Artifact> transitiveImports) {
      this.protosInDirectDependencies = protosInDirectDependencies;
      this.transitiveImports = transitiveImports;
    }

    @Override
    public Iterable<String> argv() {
      ImmutableList.Builder<String> builder = ImmutableList.builder();
      for (Artifact artifact : transitiveImports) {
        builder.add(
            "-I"
                + artifact
                    .getRootRelativePath()
                    .relativeTo(
                        artifact
                            .getOwnerLabel()
                            .getPackageIdentifier()
                            .getRepository()
                            .getPathUnderExecRoot())
                + "="
                + artifact.getExecPathString());
      }
      if (protosInDirectDependencies != null) {
        ArrayList<String> rootRelativePaths = new ArrayList<>();
        for (Artifact directDependency : protosInDirectDependencies) {
          rootRelativePaths.add(directDependency.getRootRelativePathString());
        }
        builder.add("--direct_dependencies=" + Joiner.on(":").join(rootRelativePaths));
      }
      return builder.build();
    }
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
      NestedSet<Artifact> transitiveDescriptorSets) {
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
            ruleContext.getLabel().getCanonicalForm(),
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
            "--descriptor_set_out=$(OUT)",
            null /* pluginExecutable */,
            null /* runtime */,
            NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER) /* blacklistedProtos */),
        outReplacement);
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
      String ruleLabel,
      Iterable<Artifact> outputs,
      String flavorName,
      boolean allowServices) {
    SpawnAction.Builder actions =
        createActions(
            ruleContext,
            toolchainInvocations,
            protosToCompile,
            transitiveSources,
            protosInDirectDeps,
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
      String ruleLabel,
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
        .useParameterFile(ParameterFile.ParameterFileType.UNQUOTED)
        .addOutputs(outputs)
        .setResources(GENPROTO_RESOURCE_SET)
        .useDefaultShellEnvironment()
        .setExecutable(compilerTarget)
        .setCommandLine(
            createCommandLineFromToolchains(
                toolchainInvocations,
                protosToCompile,
                transitiveSources,
                areDepsStrict(ruleContext) ? protosInDirectDeps : null,
                ruleLabel,
                allowServices,
                ruleContext.getFragment(ProtoConfiguration.class).protocOpts()))
        .setProgressMessage("Generating " + flavorName + " proto_library " + ruleContext.getLabel())
        .setMnemonic(MNEMONIC);

    return result;
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
      @Nullable NestedSet<Artifact> protosInDirectDeps,
      String ruleLabel,
      boolean allowServices,
      ImmutableList<String> protocOpts) {
    CustomCommandLine.Builder cmdLine = CustomCommandLine.builder();

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

      cmdLine.add(
          new LazyCommandLineExpansion(
              toolchain.commandLine(),
              ImmutableMap.of(
                  "OUT",
                  invocation.outReplacement,
                  "PLUGIN_OUT",
                  String.format("PLUGIN_%s_out", invocation.name))));

      if (toolchain.pluginExecutable() != null) {
        cmdLine.add(
            String.format(
                "--plugin=protoc-gen-%s=%s",
                String.format("PLUGIN_%s", invocation.name),
                toolchain.pluginExecutable().getExecutable().getExecPathString()));
      }
    }

    cmdLine.add(protocOpts);

    // Add include maps
    cmdLine.add(new ProtoCommandLineArgv(protosInDirectDeps, transitiveSources));

    if (protosInDirectDeps != null) {
      cmdLine.add(createStrictProtoDepsViolationErrorMessage(ruleLabel));
    }

    for (Artifact src : protosToCompile) {
      cmdLine.addPath(src.getRootRelativePath());
    }

    if (!allowServices) {
      cmdLine.add("--disallow_services");
    }

    return cmdLine.build();
  }

  @SuppressWarnings("FormatString") // Errorprone complains that there's no '%s' in the format
  // string, but it's actually in MESSAGE.
  @VisibleForTesting
  public static String createStrictProtoDepsViolationErrorMessage(String ruleLabel) {
    return "--direct_dependencies_violation_msg="
        + String.format(StrictProtoDepsViolationMessage.MESSAGE, ruleLabel);
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
