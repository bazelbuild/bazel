// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.Expander;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap.UmbrellaHeaderStrategy;
import com.google.devtools.build.lib.rules.objc.IntermediateArtifacts.AlwaysLink;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;

/** Utility methods for Objc rules in Starlark Builtins */
@StarlarkBuiltin(name = "objc_internal", category = DocCategory.BUILTIN, documented = false)
public class ObjcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "objc_internal";

  /**
   * Converts a possibly NoneType object to the real object if it is not NoneType or returns the
   * default value if it is.
   */
  @SuppressWarnings("unchecked")
  public static <T> T convertFromNoneable(Object obj, @Nullable T defaultValue) {
    if (Starlark.UNBOUND == obj || Starlark.isNullOrNone(obj)) {
      return defaultValue;
    }
    return (T) obj;
  }

  @StarlarkMethod(
      name = "create_compilation_attributes",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public CompilationAttributes createCompilationAttributes(StarlarkRuleContext starlarkRuleContext)
      throws EvalException, InterruptedException {
    CompilationAttributes.Builder builder = new CompilationAttributes.Builder();

    CompilationAttributes.Builder.addHeadersFromRuleContext(
        builder, starlarkRuleContext.getRuleContext());
    CompilationAttributes.Builder.addIncludesFromRuleContext(
        builder, starlarkRuleContext.getRuleContext());
    CompilationAttributes.Builder.addSdkAttributesFromRuleContext(
        builder, starlarkRuleContext.getRuleContext());
    if (starlarkRuleContext.getRuleContext().attributes().has("copts")) {
      Sequence<String> copts =
          expandAndTokenize(
              starlarkRuleContext,
              "copts",
              StarlarkList.immutableCopyOf(
                  starlarkRuleContext
                      .getRuleContext()
                      .attributes()
                      .get("copts", Types.STRING_LIST)));
      CompilationAttributes.Builder.addCompileOptionsFromRuleContext(
          builder, starlarkRuleContext.getRuleContext(), copts);
    }
    CompilationAttributes.Builder.addModuleOptionsFromRuleContext(
        builder, starlarkRuleContext.getRuleContext());

    return builder.build();
  }

  /**
   * Run variable expansion and shell tokenization on a sequence of flags.
   *
   * <p>When expanding path variables (e.g. $(execpath ...)), the label can refer to any of which in
   * the {@code srcs}, {@code non_arc_srcs}, {@code hdrs} or {@code data} attributes or an output of
   * the target.
   *
   * @param starlarkRuleContext The rule context of the expansion.
   * @param attributeName The attribute of the rule tied to the expansion. Used for error reporting
   *     only.
   * @param flags The sequence of flags to expand.
   */
  @StarlarkMethod(
      name = "expand_and_tokenize",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "attr", positional = false, named = true),
        @Param(name = "flags", positional = false, defaultValue = "[]", named = true),
      })
  public Sequence<String> expandAndTokenize(
      StarlarkRuleContext starlarkRuleContext, String attributeName, Sequence<?> flags)
      throws EvalException, InterruptedException {
    if (flags.isEmpty()) {
      return Sequence.cast(flags, String.class, attributeName);
    }
    Expander expander =
        starlarkRuleContext
            .getRuleContext()
            .getExpander(
                StarlarkRuleContext.makeLabelMap(
                    ImmutableSet.copyOf(
                        Iterables.concat(
                            starlarkRuleContext.getRuleContext().getPrerequisites("srcs"),
                            starlarkRuleContext.getRuleContext().getPrerequisites("non_arc_srcs"),
                            starlarkRuleContext.getRuleContext().getPrerequisites("hdrs"),
                            starlarkRuleContext.getRuleContext().getPrerequisites("data")))))
            .withDataExecLocations();
    ImmutableList<String> expandedFlags =
        expander.tokenized(attributeName, Sequence.cast(flags, String.class, attributeName));
    return StarlarkList.immutableCopyOf(expandedFlags);
  }

  @StarlarkMethod(
      name = "create_intermediate_artifacts",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public IntermediateArtifacts createIntermediateArtifacts(
      StarlarkRuleContext starlarkRuleContext) {
    return new IntermediateArtifacts(starlarkRuleContext.getRuleContext());
  }

  @StarlarkMethod(
      name = "j2objc_create_intermediate_artifacts",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public IntermediateArtifacts j2objcCreateIntermediateArtifacts(
      StarlarkRuleContext starlarkRuleContext) {
    return new IntermediateArtifacts(
        starlarkRuleContext.getRuleContext(),
        /* archiveFileNameSuffix= */ "_j2objc",
        UmbrellaHeaderStrategy.GENERATE,
        AlwaysLink.TRUE);
  }

  @StarlarkMethod(
      name = "create_compilation_artifacts",
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = StarlarkRuleContext.class),
              @ParamType(type = NoneType.class)
            }),
      })
  public CompilationArtifacts createCompilationArtifacts(Object starlarkRuleContextObject) {
    StarlarkRuleContext starlarkRuleContext =
        convertFromNoneable(starlarkRuleContextObject, /* defaultValue= */ null);
    if (starlarkRuleContext != null) {
      return CompilationSupport.compilationArtifacts(starlarkRuleContext.getRuleContext());
    } else {
      return new CompilationArtifacts();
    }
  }

  @StarlarkMethod(
      name = "j2objc_create_compilation_artifacts",
      documented = false,
      parameters = {
        @Param(name = "srcs", positional = false, named = true),
        @Param(name = "non_arc_srcs", positional = false, named = true),
        @Param(name = "hdrs", positional = false, named = true),
        @Param(name = "intermediate_artifacts", positional = false, named = true),
      })
  public CompilationArtifacts j2objcCreateCompilationArtifacts(
      Sequence<?> srcs,
      Sequence<?> nonArcSrcs,
      Sequence<?> hdrs,
      Object intermediateArtifactsObject)
      throws EvalException {
    IntermediateArtifacts intermediateArtifacts =
        convertFromNoneable(intermediateArtifactsObject, /* defaultValue= */ null);
    return new CompilationArtifacts(
        Sequence.cast(srcs, Artifact.class, "srcs"),
        Sequence.cast(nonArcSrcs, Artifact.class, "non_arc_srcs"),
        Sequence.cast(hdrs, Artifact.class, "hdrs"),
        intermediateArtifacts);
  }

  @StarlarkMethod(
      name = "create_compilation_context",
      documented = false,
      parameters = {
        @Param(name = "public_hdrs", positional = false, defaultValue = "[]", named = true),
        @Param(name = "public_textual_hdrs", positional = false, defaultValue = "[]", named = true),
        @Param(name = "private_hdrs", positional = false, defaultValue = "[]", named = true),
        @Param(name = "providers", positional = false, defaultValue = "[]", named = true),
        @Param(
            name = "direct_cc_compilation_contexts",
            positional = false,
            defaultValue = "[]",
            named = true),
        @Param(
            name = "cc_compilation_contexts",
            positional = false,
            defaultValue = "[]",
            named = true),
        @Param(
            name = "implementation_cc_compilation_contexts",
            positional = false,
            defaultValue = "[]",
            named = true),
        @Param(name = "defines", positional = false, defaultValue = "[]", named = true),
        @Param(name = "includes", positional = false, defaultValue = "[]", named = true),
      })
  public ObjcCompilationContext createCompilationContext(
      Sequence<?> publicHdrs,
      Sequence<?> publicTextualHdrs,
      Sequence<?> privateHdrs,
      Sequence<?> providers,
      Sequence<?> directCcCompilationContexts,
      Sequence<?> ccCompilationContexts,
      Sequence<?> implementationCcCompilationContexts,
      Sequence<?> defines,
      Sequence<?> includes)
      throws InterruptedException, EvalException {
    return ObjcCompilationContext.builder()
        .addPublicHeaders(Sequence.cast(publicHdrs, Artifact.class, "public_hdrs"))
        .addPublicTextualHeaders(
            Sequence.cast(publicTextualHdrs, Artifact.class, "public_textual_hdrs"))
        .addPrivateHeaders(Sequence.cast(privateHdrs, Artifact.class, "private_hdrs"))
        .addObjcProviders(Sequence.cast(providers, ObjcProvider.class, "providers"))
        .addDirectCcCompilationContexts(
            Sequence.cast(
                directCcCompilationContexts, CcCompilationContext.class, "cc_compilation_contexts"))
        .addCcCompilationContexts(
            Sequence.cast(
                ccCompilationContexts, CcCompilationContext.class, "cc_compilation_contexts"))
        .addImplementationCcCompilationContexts(
            Sequence.cast(
                implementationCcCompilationContexts,
                CcCompilationContext.class,
                "implementation_cc_compilation_contexts"))
        .addDefines(Sequence.cast(defines, String.class, "defines"))
        .addIncludes(
            Sequence.cast(includes, String.class, "includes").stream()
                .map(PathFragment::create)
                .collect(toImmutableList()))
        .build();
  }

  @StarlarkMethod(
      name = "subtract_linking_contexts",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "linking_contexts", positional = false, defaultValue = "[]", named = true),
        @Param(
            name = "avoid_dep_linking_contexts",
            positional = false,
            defaultValue = "[]",
            named = true),
      })
  public CcLinkingContext subtractLinkingContexts(
      StarlarkRuleContext starlarkRuleContext,
      Sequence<?> linkingContexts,
      Sequence<?> avoidDepLinkingContexts)
      throws InterruptedException, EvalException {
    return MultiArchBinarySupport.ccLinkingContextSubtractSubtrees(
        starlarkRuleContext.getRuleContext(),
        Sequence.cast(linkingContexts, CcLinkingContext.class, "linking_contexts"),
        Sequence.cast(
            avoidDepLinkingContexts, CcLinkingContext.class, "avoid_dep_linking_contexts"));
  }

  @StarlarkMethod(
      name = "get_split_target_triplet",
      documented = false,
      parameters = {@Param(name = "ctx", named = true)})
  public Dict<String, StructImpl> getSplitTargetTriplet(StarlarkRuleContext starlarkRuleContext)
      throws EvalException {
    return MultiArchBinarySupport.getSplitTargetTripletFromCtads(
        starlarkRuleContext
            .getRuleContext()
            .getSplitPrerequisites(ObjcRuleClasses.CHILD_CONFIG_ATTR));
  }

  @StarlarkMethod(
      name = "get_split_build_configs",
      documented = false,
      parameters = {@Param(name = "ctx", positional = true, named = true)})
  public Dict<String, BuildConfigurationValue> getSplitBuildConfigs(
      StarlarkRuleContext starlarkRuleContext) throws EvalException {
    Map<Optional<String>, List<ConfiguredTargetAndData>> ctads =
        starlarkRuleContext
            .getRuleContext()
            .getSplitPrerequisites(ObjcRuleClasses.CHILD_CONFIG_ATTR);
    Dict.Builder<String, BuildConfigurationValue> result = Dict.builder();
    for (Optional<String> splitTransitionKey : ctads.keySet()) {
      if (!splitTransitionKey.isPresent()) {
        throw new EvalException("unexpected empty key in split transition");
      }
      result.put(
          splitTransitionKey.get(),
          Iterables.getOnlyElement(ctads.get(splitTransitionKey)).getConfiguration());
    }
    return result.buildImmutable();
  }

  @StarlarkMethod(
      name = "get_target_platform",
      documented = false,
      parameters = {@Param(name = "build_config", positional = true, named = true)})
  public ApplePlatform getTargetPlatform(BuildConfigurationValue buildConfiguration)
      throws EvalException {
    return buildConfiguration.getFragment(AppleConfiguration.class).getSingleArchPlatform();
  }
}
