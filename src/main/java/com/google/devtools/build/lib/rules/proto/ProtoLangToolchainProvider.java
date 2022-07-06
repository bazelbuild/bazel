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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.ElementType;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.syntax.Location;

// Note: AutoValue v1.4-rc1 has AutoValue.CopyAnnotations which makes it work with Starlark. No need
// to un-AutoValue this class to expose it to Starlark.
/**
 * Specifies how to generate language-specific code from .proto files. Used by LANG_proto_library
 * rules.
 */
@AutoValue
public abstract class ProtoLangToolchainProvider {
  public static final String PROVIDER_NAME = "ProtoLangToolchainInfo";
  public static final StarlarkProvider.Key starlarkProtoLangToolchainKey =
      new StarlarkProvider.Key(
          Label.parseAbsoluteUnchecked("@_builtins//:common/proto/proto_common.bzl"),
          PROVIDER_NAME);
  public static final StarlarkProviderIdentifier PROVIDER_ID =
      StarlarkProviderIdentifier.forKey(starlarkProtoLangToolchainKey);

  // Format string used when passing output to the plugin used by proto compiler.
  public abstract String outReplacementFormatFlag();

  // Format string used when passing plugin to proto compiler.
  @Nullable
  public abstract String pluginFormatFlag();

  // Proto compiler plugin.
  @Nullable
  public abstract FilesToRunProvider pluginExecutable();

  @Nullable
  public abstract TransitiveInfoCollection runtime();

  /**
   * Returns a list of {@link ProtoSource}s that are already provided by the protobuf runtime (i.e.
   * for which {@code <lang>_proto_library} should not generate bindings.
   */
  // Proto sources provided by the toolchain.
  public abstract ImmutableList<ProtoSource> providedProtoSources();

  // Proto compiler.
  public abstract FilesToRunProvider protoc();

  // Options to pass to proto compiler.
  public StarlarkList<String> protocOptsForStarlark() {
    return StarlarkList.immutableCopyOf(protocOpts());
  }

  public abstract ImmutableList<String> protocOpts();

  // Progress message to set on the proto compiler action.
  public abstract String progressMessage();

  // Mnemonic to set on the proto compiler action.
  public abstract String mnemonic();

  public static StarlarkInfo create(
      String outReplacementFormatFlag,
      String pluginFormatFlag,
      FilesToRunProvider pluginExecutable,
      TransitiveInfoCollection runtime,
      ImmutableList<ProtoSource> providedProtoSources,
      FilesToRunProvider protoc,
      ImmutableList<String> protocOpts,
      String progressMessage,
      String mnemonic) {

    NestedSetBuilder<ProtoSource> providedProtoSourcesSet = NestedSetBuilder.stableOrder();
    providedProtoSources.forEach(providedProtoSourcesSet::add);
    NestedSetBuilder<String> protocOptsSet = NestedSetBuilder.stableOrder();
    protocOpts.forEach(protocOptsSet::add);

    Map<String, Object> m = new LinkedHashMap<>();
    m.put("plugin", pluginExecutable == null ? Starlark.NONE : pluginExecutable);
    m.put("plugin_format_flag", pluginFormatFlag);
    m.put("proto_compiler", protoc == null ? Starlark.NONE : protoc);
    m.put(
        "provided_proto_sources",
        Depset.of(ElementType.of(ProtoSource.class), providedProtoSourcesSet.build()));
    m.put("protoc_opts", Depset.of(ElementType.of(ProtoSource.class), protocOptsSet.build()));
    m.put("out_replacement_format_flag", outReplacementFormatFlag);
    m.put("progress_message", progressMessage);
    m.put("mnemonic", mnemonic);
    m.put("plugin", pluginExecutable == null ? Starlark.NONE : pluginExecutable);
    m.put("runtime", runtime == null ? Starlark.NONE : runtime);

    StarlarkProvider.Builder builder =
        StarlarkProvider.builder(
            Location.fromFileLineColumn(protoc.getExecutable().getFilename(), 0, 0));
    builder.setExported(starlarkProtoLangToolchainKey);

    return StarlarkInfo.create(builder.build(), m, Location.BUILTIN);
  }

  private static ImmutableList<ProtoLangToolchainProvider> getToolchains(
      RuleContext ruleContext, String attributeName) {
    ImmutableList.Builder<ProtoLangToolchainProvider> result = ImmutableList.builder();
    for (TransitiveInfoCollection prerequisite : ruleContext.getPrerequisites(attributeName)) {
      ProtoLangToolchainProvider toolchain = get(prerequisite);
      if (toolchain != null) {
        result.add(toolchain);
      }
    }
    return result.build();
  }

  @Nullable
  public static ProtoLangToolchainProvider get(RuleContext ruleContext, String attributeName) {
    return getToolchains(ruleContext, attributeName).stream().findFirst().orElse(null);
  }

  public static ProtoLangToolchainProvider get(TransitiveInfoCollection prerequisite) {
    StarlarkInfo provider = (StarlarkInfo) prerequisite.get(starlarkProtoLangToolchainKey);
    return wrapStarlarkProviderWithNativeProvider(provider);
  }

  @Nullable
  public static StarlarkInfo getStarlarkProvider(RuleContext ruleContext, String attributeName) {
    for (TransitiveInfoCollection prerequisite : ruleContext.getPrerequisites(attributeName)) {
      StarlarkInfo provider = (StarlarkInfo) prerequisite.get(starlarkProtoLangToolchainKey);
      if (provider != null) {
        return provider;
      }
    }
    return null;
  }

  public static StarlarkInfo getStarlarkProvider(TransitiveInfoCollection prerequisite) {
    return (StarlarkInfo) prerequisite.get(starlarkProtoLangToolchainKey);
  }

  @Nullable
  @SuppressWarnings("unchecked")
  private static ProtoLangToolchainProvider wrapStarlarkProviderWithNativeProvider(
      StarlarkInfo provider) {
    if (provider != null) {
      try {
        return new AutoValue_ProtoLangToolchainProvider(
            provider.getValue("out_replacement_format_flag", String.class),
            provider.getValue("plugin_format_flag", String.class),
            provider.getValue("plugin") instanceof NoneType
                ? null
                : provider.getValue("plugin", FilesToRunProvider.class),
            provider.getValue("runtime") instanceof NoneType
                ? null
                : provider.getValue("runtime", TransitiveInfoCollection.class),
            ImmutableList.copyOf(
                (StarlarkList<ProtoSource>) provider.getValue("provided_proto_sources")),
            provider.getValue("proto_compiler", FilesToRunProvider.class),
            ImmutableList.copyOf((StarlarkList<String>) provider.getValue("protoc_opts")),
            provider.getValue("progress_message", String.class),
            provider.getValue("mnemonic", String.class));
      } catch (EvalException e) {
        return null;
      }
    }
    return null;
  }
}
