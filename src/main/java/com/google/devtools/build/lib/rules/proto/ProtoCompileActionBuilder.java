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

import static com.google.common.collect.Iterables.isEmpty;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.ResourceSetOrBuilder;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.starlark.Args;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.ElementType;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.util.OS;
import java.util.HashSet;
import java.util.List;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/** Constructs actions to run the protocol compiler to generate sources from .proto files. */
public class ProtoCompileActionBuilder {
  private static final String DEFAULT_MNEMONIC = "GenProto";

  @VisibleForTesting
  public static final String STRICT_DEPS_FLAG_TEMPLATE =
      "--direct_dependencies_violation_msg=" + ProtoConstants.STRICT_PROTO_DEPS_VIOLATION_MESSAGE;

  private final ProtoInfo protoInfo;
  private final FilesToRunProvider protoCompiler;
  private final String progressMessage;
  private final Iterable<Artifact> outputs;
  private Iterable<Artifact> inputs;
  private FilesToRunProvider langPlugin;
  private String langPluginFormat;
  private Iterable<String> langPluginParameter;
  private String langPluginParameterFormat;
  private boolean hasServices;
  private Iterable<String> additionalCommandLineArguments;
  private Iterable<FilesToRunProvider> additionalTools;
  private boolean checkStrictImportPublic;
  private String mnemonic;

  public ProtoCompileActionBuilder allowServices(boolean hasServices) {
    this.hasServices = hasServices;
    return this;
  }

  public ProtoCompileActionBuilder setInputs(Iterable<Artifact> inputs) {
    this.inputs = inputs;
    return this;
  }

  public ProtoCompileActionBuilder setLangPlugin(
      FilesToRunProvider langPlugin, String langPluginFormat) {
    this.langPlugin = langPlugin;
    this.langPluginFormat = langPluginFormat;
    return this;
  }

  public ProtoCompileActionBuilder setMnemonic(String mnemonic) {
    this.mnemonic = mnemonic;
    return this;
  }

  public ProtoCompileActionBuilder setLangPluginParameter(
      Iterable<String> langPluginParameter, String langPluginParameterFormat) {
    this.langPluginParameter = langPluginParameter;
    this.langPluginParameterFormat = langPluginParameterFormat;
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
      ProtoInfo protoInfo,
      FilesToRunProvider protoCompiler,
      String progressMessage,
      Iterable<Artifact> outputs) {
    this.protoInfo = protoInfo;
    this.protoCompiler = protoCompiler;
    this.progressMessage = progressMessage;
    this.outputs = outputs;
    this.mnemonic = DEFAULT_MNEMONIC;
  }

  /** Builds a ResourceSet based on the number of inputs. */
  public static class ProtoCompileResourceSetBuilder implements ResourceSetOrBuilder {
    @Override
    public ResourceSet buildResourceSet(OS os, int inputsSize) {
      return ResourceSet.createWithRamCpu(
          /* memoryMb= */ 25 + 0.15 * inputsSize, /* cpuUsage= */ 1);
    }
  }

  public void maybeRegister(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    if (isEmpty(outputs)) {
      return;
    }

    ruleContext.initStarlarkRuleContext();
    StarlarkThread thread = ruleContext.getStarlarkThread();
    Args additionalArgs = Args.newArgs(thread.mutability(), thread.getSemantics());

    try {
      if (langPlugin != null && langPlugin.getExecutable() != null) {
        // We pass a separate langPlugin as there are plugins that cannot be overridden
        // and thus we have to deal with "$xx_plugin" and "xx_plugin".
        additionalArgs.addArgument(
            langPlugin.getExecutable(), /*value=*/ Starlark.UNBOUND, langPluginFormat, thread);
      }

      if (langPluginParameter != null) {
        additionalArgs.addJoined(
            StarlarkList.immutableCopyOf(langPluginParameter),
            /*values=*/ Starlark.UNBOUND,
            /*joinWith=*/ "",
            /*mapEach=*/ Starlark.NONE,
            /*formatEach=*/ Starlark.NONE,
            /*formatJoined=*/ langPluginParameterFormat,
            /*omitIfEmpty=*/ true,
            /*uniquify=*/ false,
            /*expandDirectories=*/ true,
            /*allowClosure=*/ false,
            thread);
      }

      if (!hasServices) {
        additionalArgs.addArgument(
            "--disallow_services",
            /* value = */ Starlark.UNBOUND,
            /* format = */ Starlark.NONE,
            thread);
      }

      if (additionalCommandLineArguments != null) {
        additionalArgs.addAll(
            StarlarkList.immutableCopyOf(additionalCommandLineArguments),
            /*values=*/ Starlark.UNBOUND,
            /*mapEach=*/ Starlark.NONE,
            /*formatEach=*/ Starlark.NONE,
            /*beforeEach=*/ Starlark.NONE,
            /*omitIfEmpty=*/ true,
            /*uniquify=*/ false,
            /*expandDirectories=*/ true,
            /*terminateWith=*/ Starlark.NONE,
            /*allowClosure=*/ false,
            thread);
      }
    } catch (EvalException e) {
      throw ruleContext.throwWithRuleError(e);
    }

    ImmutableList.Builder<FilesToRunProvider> plugins = new ImmutableList.Builder<>();
    if (additionalTools != null) {
      plugins.addAll(additionalTools);
    }
    if (langPlugin != null) {
      plugins.add(langPlugin);
    }

    StarlarkFunction createProtoCompileAction =
        (StarlarkFunction) ruleContext.getStarlarkDefinedBuiltin("create_proto_compile_action");

    ruleContext.callStarlarkOrThrowRuleError(
        createProtoCompileAction,
        ImmutableList.of(
            /* ctx */ ruleContext.getStarlarkRuleContext(),
            /* proto_info */ protoInfo,
            /* proto_compiler */ protoCompiler,
            /* progress_message */ progressMessage,
            /* outputs */ StarlarkList.immutableCopyOf(outputs),
            /* additional_args */ additionalArgs,
            /* plugins */ StarlarkList.immutableCopyOf(plugins.build()),
            /* mnemonic */ mnemonic,
            /* strict_imports */ checkStrictImportPublic,
            /* additional_inputs */ inputs == null
                ? Depset.of(ElementType.EMPTY, NestedSetBuilder.emptySet(Order.STABLE_ORDER))
                : Depset.of(Artifact.TYPE, NestedSetBuilder.wrap(Order.STABLE_ORDER, inputs)),
            /* resource_set */
            new StarlarkCallable() {
              @Override
              public String getName() {
                return "proto_compile_resource_set";
              }

              @Override
              public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) {
                // args are a tuple of OS and inputsSize
                int inputsSize = ((StarlarkInt) args.get(1)).toIntUnchecked();
                return Dict.immutableCopyOf(
                    ImmutableMap.of(
                        "memory",
                        StarlarkFloat.of(25 + 0.15 * inputsSize),
                        "cpu",
                        StarlarkInt.of(1)));
              }
            }),
        ImmutableMap.of());
  }

  /** Whether to allow services in the proto compiler invocation. */
  public enum Services {
    ALLOW,
    DISALLOW,
  }

  /**
   * Registers actions to generate code from .proto files.
   *
   * <p>This method uses information from proto_lang_toolchain() rules. New rules should use this
   * method instead of the soup of methods above.
   *
   * @param outputs The artifacts that the resulting action must create.
   * @param progressMessage Please use "Generating {flavorName} proto_library %{label}".
   * @param allowServices If false, the compilation will break if any .proto file has service
   */
  public static void registerActions(
      RuleContext ruleContext,
      List<ToolchainInvocation> toolchainInvocations,
      ProtoInfo protoInfo,
      Iterable<Artifact> outputs,
      String progressMessage,
      Services allowServices)
      throws RuleErrorException, InterruptedException {
    if (isEmpty(outputs)) {
      return;
    }

    ProtoToolchainInfo protoToolchain = ProtoToolchainInfo.fromRuleContext(ruleContext);
    if (protoToolchain == null) {
      return;
    }

    ruleContext.initStarlarkRuleContext();
    StarlarkThread thread = ruleContext.getStarlarkThread();
    Args additionalArgs = Args.newArgs(thread.mutability(), thread.getSemantics());

    // A set to check if there are multiple invocations with the same name.
    HashSet<String> invocationNames = new HashSet<>();
    ImmutableList.Builder<Object> plugins = ImmutableList.builder();

    try {
      for (ToolchainInvocation invocation : toolchainInvocations) {
        if (!invocationNames.add(invocation.name)) {
          throw new IllegalStateException(
              "Invocation name "
                  + invocation.name
                  + " appears more than once. "
                  + "This could lead to incorrect proto-compiler behavior");
        }

        ProtoLangToolchainProvider toolchain = invocation.toolchain;

        String format = toolchain.outReplacementFormatFlag();
        additionalArgs.addArgument(
            invocation.outReplacement, /*value=*/ Starlark.UNBOUND, format, thread);

        if (toolchain.pluginExecutable() != null) {
          additionalArgs.addArgument(
              toolchain.pluginExecutable().getExecutable(),
              /*value=*/ Starlark.UNBOUND,
              toolchain.pluginFormatFlag(),
              thread);
          plugins.add(toolchain.pluginExecutable());
        }

        additionalArgs.addJoined(
            StarlarkList.immutableCopyOf(invocation.protocOpts),
            /*values=*/ Starlark.UNBOUND,
            /*joinWith=*/ "",
            /*mapEach=*/ Starlark.NONE,
            /*formatEach=*/ Starlark.NONE,
            /*formatJoined=*/ Starlark.NONE,
            /*omitIfEmpty=*/ true,
            /*uniquify=*/ false,
            /*expandDirectories=*/ true,
            /*allowClosure=*/ false,
            thread);
      }

      if (allowServices == Services.DISALLOW) {
        additionalArgs.addArgument(
            "--disallow_services", /*value=*/ Starlark.UNBOUND, /*format=*/ Starlark.NONE, thread);
      }
    } catch (EvalException e) {
      throw ruleContext.throwWithRuleError(e.getMessageWithStack());
    }

    StarlarkFunction createProtoCompileAction =
        (StarlarkFunction) ruleContext.getStarlarkDefinedBuiltin("create_proto_compile_action");
    ruleContext.callStarlarkOrThrowRuleError(
        createProtoCompileAction,
        ImmutableList.of(
            /* ctx */ ruleContext.getStarlarkRuleContext(),
            /* proto_info */ protoInfo,
            /* proto_compiler */ protoToolchain.getCompiler(),
            /* progress_message */ progressMessage,
            /* outputs */ StarlarkList.immutableCopyOf(outputs),
            /* additional_args */ additionalArgs,
            /* plugins */ StarlarkList.immutableCopyOf(plugins.build())),
        ImmutableMap.of());
  }

  /**
   * Describes a toolchain and the value to replace for a $(OUT) that might appear in its
   * commandLine() (e.g., "bazel-out/foo.srcjar").
   */
  public static class ToolchainInvocation {
    final String name;
    public final ProtoLangToolchainProvider toolchain;
    final CharSequence outReplacement;
    final ImmutableList<String> protocOpts;

    public ToolchainInvocation(
        String name, ProtoLangToolchainProvider toolchain, CharSequence outReplacement) {
      this(name, toolchain, outReplacement, ImmutableList.of());
    }

    public ToolchainInvocation(
        String name,
        ProtoLangToolchainProvider toolchain,
        CharSequence outReplacement,
        ImmutableList<String> protocOpts) {
      Preconditions.checkState(!name.contains(" "), "Name %s should not contain spaces", name);
      this.name = name;
      this.toolchain = toolchain;
      this.outReplacement = outReplacement;
      this.protocOpts = Preconditions.checkNotNull(protocOpts);
    }
  }
}
