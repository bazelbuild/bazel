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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.starlark.Args;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.ElementType;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/** Utility functions for proto_library and proto aspect implementations. */
public class ProtoCommon {
  private ProtoCommon() {
    throw new UnsupportedOperationException();
  }

  // Keep in sync with the migration label in
  // https://github.com/bazelbuild/rules_proto/blob/master/proto/defs.bzl.
  @VisibleForTesting
  public static final String PROTO_RULES_MIGRATION_LABEL =
      "__PROTO_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__";

  private static final Interner<PathFragment> PROTO_SOURCE_ROOT_INTERNER =
      BlazeInterners.newWeakInterner();

  /**
   * Returns a memory efficient version of the passed protoSourceRoot.
   *
   * <p>Any sizable proto graph will contain many {@code .proto} sources with the same source root.
   * We can't afford to have all of them represented as individual objects in memory.
   *
   * @param protoSourceRoot
   * @return
   */
  static PathFragment memoryEfficientProtoSourceRoot(PathFragment protoSourceRoot) {
    return PROTO_SOURCE_ROOT_INTERNER.intern(protoSourceRoot);
  }

  // =================================================================
  // Protocol compiler invocation stuff.

  /**
   * Decides whether this proto_library should check for strict proto deps.
   *
   * <p>Only takes into account the command-line flag --strict_proto_deps.
   */
  @VisibleForTesting
  public static boolean areDepsStrict(RuleContext ruleContext) {
    StrictDepsMode getBool = ruleContext.getFragment(ProtoConfiguration.class).strictProtoDeps();
    return getBool != StrictDepsMode.OFF && getBool != StrictDepsMode.DEFAULT;
  }

  public static void checkPrivateStarlarkificationAllowlist(StarlarkThread thread)
      throws EvalException {
    Label label =
        ((BazelModuleContext) Module.ofInnermostEnclosingStarlarkFunction(thread).getClientData())
            .label();
    if (!label.getPackageIdentifier().getRepository().toString().equals("@_builtins")) {
      throw Starlark.errorf("Rule in '%s' cannot use private API", label.getPackageName());
    }
  }

  public static ImmutableList<Artifact> declareGeneratedFiles(
      RuleContext ruleContext, ConfiguredTarget protoTarget, String extension)
      throws RuleErrorException, InterruptedException {
    StarlarkFunction declareGeneratedFiles =
        (StarlarkFunction)
            ruleContext.getStarlarkDefinedBuiltin("proto_common_declare_generated_files");
    ruleContext.initStarlarkRuleContext();
    Sequence<?> outputs =
        (Sequence<?>)
            ruleContext.callStarlarkOrThrowRuleError(
                declareGeneratedFiles,
                ImmutableList.of(
                    /* actions */ ruleContext.getStarlarkRuleContext().actions(),
                    /* proto_library_target */ Starlark.NONE,
                    /* extension */ extension),
                ImmutableMap.of("proto_info", protoTarget.get(ProtoInfo.PROVIDER)));
    try {
      return Sequence.cast(outputs, Artifact.class, "declare_generated_files").getImmutableList();
    } catch (EvalException e) {
      throw new RuleErrorException(e.getMessageWithStack());
    }
  }

  private static final StarlarkCallable pythonMapper =
      new StarlarkCallable() {
        @Override
        public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) {
          return args.get(0).toString().replace('-', '_').replace('.', '/');
        }

        @Override
        public String getName() {
          return "python_mapper";
        }
      };

  /**
   * Each language-specific initialization method will call this to construct Artifacts representing
   * its protocol compiler outputs. The cals replaces hyphens in the file name with underscores, and
   * dots in the file name with forward slashes, as required for Python modules.
   *
   * @param extension Remove ".proto" and replace it with this to produce the output file name, e.g.
   *     ".pb.cc".
   */
  public static ImmutableList<Artifact> declareGeneratedFilesPython(
      RuleContext ruleContext, ConfiguredTarget protoTarget, String extension)
      throws RuleErrorException, InterruptedException {
    StarlarkFunction declareGeneratedFiles =
        (StarlarkFunction)
            ruleContext.getStarlarkDefinedBuiltin("proto_common_declare_generated_files");
    ruleContext.initStarlarkRuleContext();
    Sequence<?> outputs =
        (Sequence<?>)
            ruleContext.callStarlarkOrThrowRuleError(
                declareGeneratedFiles,
                ImmutableList.of(
                    /* actions */ ruleContext.getStarlarkRuleContext().actions(),
                    /* proto_library_target */ Starlark.NONE,
                    /* extension */ extension,
                    /* experimental_python_names */ pythonMapper),
                ImmutableMap.of("proto_info", protoTarget.get(ProtoInfo.PROVIDER)));
    try {
      return Sequence.cast(outputs, Artifact.class, "declare_generated_files").getImmutableList();
    } catch (EvalException e) {
      throw new RuleErrorException(e.getMessageWithStack());
    }
  }

  public static void compile(
      RuleContext ruleContext,
      ConfiguredTarget protoTarget,
      StarlarkInfo protoLangToolchainInfo,
      Iterable<Artifact> generatedFiles,
      @Nullable Object pluginOutput,
      @Nullable Args additionalArgs,
      Iterable<FilesToRunProvider> additionalTools,
      Iterable<Artifact> additionalInputs,
      @Nullable StarlarkCallable resourceSet,
      String progressMessage)
      throws RuleErrorException, InterruptedException {
    StarlarkFunction compile =
        (StarlarkFunction) ruleContext.getStarlarkDefinedBuiltin("proto_common_compile");
    ruleContext.initStarlarkRuleContext();
    ruleContext.callStarlarkOrThrowRuleError(
        compile,
        ImmutableList.of(
            /* actions */ ruleContext.getStarlarkRuleContext().actions(),
            /* proto_lang_toolchain_info */ protoLangToolchainInfo,
            /* generated_files */ StarlarkList.immutableCopyOf(generatedFiles),
            /* plugin_output */ pluginOutput == null ? Starlark.NONE : pluginOutput,
            /* additional_args */ additionalArgs == null ? Starlark.NONE : additionalArgs,
            /* additional_tools */ StarlarkList.immutableCopyOf(additionalTools),
            /* additional_inputs */ additionalInputs == null
                ? Depset.of(ElementType.EMPTY, NestedSetBuilder.emptySet(Order.STABLE_ORDER))
                : Depset.of(
                    Artifact.TYPE, NestedSetBuilder.wrap(Order.STABLE_ORDER, additionalInputs)),
            /* resource_set */ resourceSet == null ? Starlark.NONE : resourceSet,
            /* experimental_progress_message */ progressMessage,
            /* proto_info */ protoTarget.get(ProtoInfo.PROVIDER)),
        ImmutableMap.of());
  }

  public static void compile(
      RuleContext ruleContext,
      ConfiguredTarget protoTarget,
      StarlarkInfo protoLangToolchainInfo,
      Iterable<Artifact> generatedFiles,
      @Nullable Object pluginOutput,
      String progressMessage)
      throws RuleErrorException, InterruptedException {
    StarlarkFunction compile =
        (StarlarkFunction) ruleContext.getStarlarkDefinedBuiltin("proto_common_compile");
    ruleContext.initStarlarkRuleContext();
    ruleContext.callStarlarkOrThrowRuleError(
        compile,
        ImmutableList.of(
            /* actions */ ruleContext.getStarlarkRuleContext().actions(),
            /* proto_lang_toolchain_info */ protoLangToolchainInfo,
            /* generated_files */ StarlarkList.immutableCopyOf(generatedFiles),
            /* plugin_output */ pluginOutput == null ? Starlark.NONE : pluginOutput),
        ImmutableMap.of(
            "experimental_progress_message",
            progressMessage,
            "proto_info",
            protoTarget.get(ProtoInfo.PROVIDER)));
  }

  public static boolean shouldGenerateCode(
      RuleContext ruleContext,
      ConfiguredTarget protoTarget,
      StarlarkInfo protoLangToolchainInfo,
      String ruleName)
      throws RuleErrorException, InterruptedException {
    StarlarkFunction shouldGenerateCode =
        (StarlarkFunction)
            ruleContext.getStarlarkDefinedBuiltin("proto_common_experimental_should_generate_code");
    ruleContext.initStarlarkRuleContext();
    return (Boolean)
        ruleContext.callStarlarkOrThrowRuleError(
            shouldGenerateCode,
            ImmutableList.of(
                /* proto_info */ protoTarget.get(ProtoInfo.PROVIDER),
                /* proto_lang_toolchain_info */ protoLangToolchainInfo,
                /* rule_name */ ruleName,
                /* target_label */ protoTarget.getLabel()),
            ImmutableMap.of());
  }

  public static Sequence<Artifact> filterSources(
      RuleContext ruleContext, ConfiguredTarget protoTarget, StarlarkInfo protoLangToolchainInfo)
      throws RuleErrorException, InterruptedException {
    StarlarkFunction filterSources =
        (StarlarkFunction)
            ruleContext.getStarlarkDefinedBuiltin("proto_common_experimental_filter_sources");
    ruleContext.initStarlarkRuleContext();
    try {
      return Sequence.cast(
          ((Tuple)
                  ruleContext.callStarlarkOrThrowRuleError(
                      filterSources,
                      ImmutableList.of(
                          /* proto_info */ protoTarget.get(ProtoInfo.PROVIDER),
                          /* proto_lang_toolchain_info */ protoLangToolchainInfo),
                      ImmutableMap.of()))
              .get(0),
          Artifact.class,
          "included");
    } catch (EvalException e) {

      throw new RuleErrorException(e.getMessageWithStack());
    }
  }
}
