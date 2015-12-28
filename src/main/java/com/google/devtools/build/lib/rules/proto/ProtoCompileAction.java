// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Optional.absent;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.Iterables.isEmpty;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.CustomMultiArgv;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.util.LazyString;

import java.util.List;

/**
 * An action to run the protocol compiler to generate sources from .proto files.
 */
public final class ProtoCompileAction {

  private static final String MNEMONIC = "GenProto";
  private static final ResourceSet GENPROTO_RESOURCE_SET =
      ResourceSet.createWithRamCpuIo(100, .1, .0);

  private final RuleContext ruleContext;
  private final SupportData supportData;
  private final String language;
  private final Iterable<Artifact> outputs;
  private final List<? extends CharSequence> prefixArguments;
  private final FilesToRunProvider langPluginTarget;
  private final FilesToRunProvider compilerTarget;
  private final List<String> suffixArguments;

  public static class Builder {
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

    public Builder setRuleContext(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
      return this;
    }

    public Builder setSupportData(SupportData supportData) {
      this.supportData = supportData;
      return this;
    }

    public Builder setLanguage(String language) {
      this.language = language;
      return this;
    }

    public Builder setLangPrefix(String langPrefix) {
      this.langPrefix = langPrefix;
      return this;
    }

    public Builder setHasServices(boolean hasServices) {
      this.hasServices = hasServices;
      return this;
    }

    public Builder setOutputs(Iterable<Artifact> outputs) {
      this.outputs = outputs;
      return this;
    }

    public Builder setLangParameter(String langParameter) {
      this.langParameter = langParameter;
      return this;
    }

    public Builder setLangPluginName(String langPluginName) {
      this.langPluginName = langPluginName;
      return this;
    }

    public Builder setLangPluginParameter(String langPluginParameter) {
      this.langPluginParameter = langPluginParameter;
      return this;
    }

    public Builder setLangPluginParameterSupplier(Supplier<String> langPluginParameterSupplier) {
      this.langPluginParameterSupplier = langPluginParameterSupplier;
      return this;
    }

    public Builder(RuleContext ruleContext, SupportData supportData, String language,
        String langPrefix, Iterable<Artifact> outputs) {
      this.ruleContext = ruleContext;
      this.supportData = supportData;
      this.language = language;
      this.langPrefix = langPrefix;
      this.outputs = outputs;
    }

    public Optional<ProtoCompileAction> build() {
      checkState(langPluginParameter == null || langPluginParameterSupplier == null,
          "Only one of {langPluginParameter, langPluginParameterSupplier} should be set.");

      final Supplier<String> langPluginParameter1 =
          langPluginParameter == null
              ? langPluginParameterSupplier
              : Suppliers.ofInstance(langPluginParameter);
      if (isEmpty(outputs)) {
        return absent();
      }

      FilesToRunProvider langPluginTarget = null;
      List<? extends CharSequence> prefixArguments;
      if (langPluginName != null) {
        Preconditions.checkArgument(langParameter == null);
        Preconditions.checkArgument(langPluginParameter1 != null);
        // We pass a separate langPluginName as there are plugins that cannot be overridden
        // and thus we have to deal with "$xx_plugin" and "xx_plugin".
        langPluginTarget = ruleContext.getExecutablePrerequisite(langPluginName, Mode.HOST);
        if (ruleContext.hasErrors()) {
          return absent();
        }
        LazyString lazyLangPlugingFlag =
            new LazyString() {
              @Override
              public String toString() {
                return String.format("--%s_out=%s", langPrefix, langPluginParameter1.get());
              }
            };
        prefixArguments =
            ImmutableList.of(
                String.format(
                    "--plugin=protoc-gen-%s=%s",
                    langPrefix,
                    langPluginTarget.getExecutable().getExecPathString()),
                lazyLangPlugingFlag);
      } else {
        prefixArguments =
            (langParameter != null) ? ImmutableList.of(langParameter) : ImmutableList.<String>of();
      }

      List<String> suffixArguments =
          hasServices ? ImmutableList.<String>of() : ImmutableList.of("--disallow_services");

      FilesToRunProvider compilerTarget =
          ruleContext.getExecutablePrerequisite("$compiler", Mode.HOST);

      if (ruleContext.hasErrors()) {
        return absent();
      }

      return Optional.of(
          new ProtoCompileAction(
              ruleContext,
              supportData,
              language,
              suffixArguments,
              outputs,
              prefixArguments,
              langPluginTarget,
              compilerTarget));
    }
  }

  /**
   * A convenience method to register an action, if it's present.
   * @param protoCompileActionOptional
   */
  public static void registerAction(Optional<ProtoCompileAction> protoCompileActionOptional) {
    if (protoCompileActionOptional.isPresent()) {
      protoCompileActionOptional.get().registerAction();
    }
  }

  public ProtoCompileAction(
      RuleContext ruleContext,
      SupportData supportData,
      String language,
      List<String> suffixArguments,
      Iterable<Artifact> outputs,
      List<? extends CharSequence> prefixArguments,
      FilesToRunProvider langPluginTarget,
      FilesToRunProvider compilerTarget) {
    this.ruleContext = ruleContext;
    this.supportData = supportData;
    this.language = language;
    this.suffixArguments = suffixArguments;
    this.outputs = outputs;
    this.prefixArguments = prefixArguments;
    this.langPluginTarget = langPluginTarget;
    this.compilerTarget = compilerTarget;
  }

  /**
   * Registers a proto compile action with the RuleContext.
   */
  public void registerAction() {
    SpawnAction.Builder action = createAction(protoCompileCommandLine().build());
    ruleContext.registerAction(action.build(ruleContext));
  }

  public SpawnAction.Builder createAction(CommandLine commandLine) {
    SpawnAction.Builder builder =
        new SpawnAction.Builder().addTransitiveInputs(supportData.getTransitiveImports());

    // We also depend on the strict protodeps result to ensure this is run.
    if (supportData.getUsedDirectDeps() != null) {
      builder.addInput(supportData.getUsedDirectDeps());
    }

    if (langPluginTarget != null) {
      builder.addTool(langPluginTarget);
    }

    builder
        .addOutputs(outputs)
        .setResources(GENPROTO_RESOURCE_SET)
        .useDefaultShellEnvironment()
        .setExecutable(compilerTarget)
        .setCommandLine(commandLine)
        .setProgressMessage("Generating " + language + " proto_library " + ruleContext.getLabel())
        .setMnemonic(MNEMONIC);

    return builder;
  }

  /* Commandline generator for protoc invocations. */
  public CustomCommandLine.Builder protoCompileCommandLine() {
    CustomCommandLine.Builder arguments = CustomCommandLine.builder();
    for (CharSequence charSequence : prefixArguments) {
      arguments.add(charSequence);
    }
    arguments.add(ruleContext.getFragment(ProtoConfiguration.class).protocOpts());

    // Add include maps
    arguments.add(
        new CustomMultiArgv() {
          @Override
          public Iterable<String> argv() {
            ImmutableList.Builder<String> builder = ImmutableList.builder();
            for (Artifact artifact : supportData.getTransitiveImports()) {
              builder.add(
                  "-I"
                      + artifact.getRootRelativePath().getPathString()
                      + "="
                      + artifact.getExecPathString());
            }
            return builder.build();
          }
        });

    for (Artifact src : supportData.getDirectProtoSources()) {
      arguments.addPath(src.getRootRelativePath());
    }

    arguments.add(suffixArguments);
    return arguments;
  }
}

