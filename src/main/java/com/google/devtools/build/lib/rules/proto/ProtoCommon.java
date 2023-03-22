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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/** Utility functions for proto_library and proto aspect implementations. */
public class ProtoCommon {
  private ProtoCommon() {
    throw new UnsupportedOperationException();
  }

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
                    /* proto_info */ protoTarget.get(ProtoInfo.PROVIDER.getKey()),
                    /* extension */ extension),
                ImmutableMap.of());
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
      String progressMessage,
      String execGroup)
      throws RuleErrorException, InterruptedException {
    StarlarkFunction compile =
        (StarlarkFunction) ruleContext.getStarlarkDefinedBuiltin("proto_common_compile");
    ruleContext.initStarlarkRuleContext();
    ruleContext.callStarlarkOrThrowRuleError(
        compile,
        ImmutableList.of(
            /* actions */ ruleContext.getStarlarkRuleContext().actions(),
            /* proto_info */ protoTarget.get(ProtoInfo.PROVIDER.getKey()),
            /* proto_lang_toolchain_info */ protoLangToolchainInfo,
            /* generated_files */ StarlarkList.immutableCopyOf(generatedFiles),
            /* plugin_output */ pluginOutput == null ? Starlark.NONE : pluginOutput),
        ImmutableMap.of(
            "experimental_progress_message",
            progressMessage,
            "experimental_exec_group",
            execGroup));
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
                          /* proto_info */ protoTarget.get(ProtoInfo.PROVIDER.getKey()),
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
