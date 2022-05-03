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
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkList;

// Note: AutoValue v1.4-rc1 has AutoValue.CopyAnnotations which makes it work with Starlark. No need
// to un-AutoValue this class to expose it to Starlark.
/**
 * Specifies how to generate language-specific code from .proto files. Used by LANG_proto_library
 * rules.
 */
@AutoValue
public abstract class ProtoLangToolchainProvider extends NativeInfo {
  public static final String PROVIDER_NAME = "ProtoLangToolchainInfo";
  public static final Provider PROVIDER = new Provider();

  /** Provider class for {@link ProtoLangToolchainProvider} objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  public static class Provider extends BuiltinProvider<ProtoLangToolchainProvider> {
    public Provider() {
      super(PROVIDER_NAME, ProtoLangToolchainProvider.class);
    }
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @StarlarkMethod(
      name = "out_replacement_format_flag",
      doc = "Format string used when passing output to the plugin used by proto compiler.",
      structField = true)
  public abstract String outReplacementFormatFlag();

  @StarlarkMethod(
      name = "plugin_format_flag",
      doc = "Format string used when passing plugin to proto compiler.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  public abstract String pluginFormatFlag();

  @StarlarkMethod(
      name = "plugin",
      doc = "Proto compiler plugin.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  public abstract FilesToRunProvider pluginExecutable();

  @Nullable
  public abstract TransitiveInfoCollection runtime();

  /**
   * Returns a list of {@link ProtoSource}s that are already provided by the protobuf runtime (i.e.
   * for which {@code <lang>_proto_library} should not generate bindings.
   */
  @StarlarkMethod(
      name = "provided_proto_sources",
      doc = "Proto sources provided by the toolchain.",
      structField = true)
  public abstract ImmutableList<ProtoSource> providedProtoSources();

  @StarlarkMethod(name = "proto_compiler", doc = "Proto compiler.", structField = true)
  public abstract FilesToRunProvider protoc();

  @StarlarkMethod(
      name = "protoc_opts",
      doc = "Options to pass to proto compiler.",
      structField = true)
  public StarlarkList<String> protocOptsForStarlark() {
    return StarlarkList.immutableCopyOf(protocOpts());
  }

  public abstract ImmutableList<String> protocOpts();

  @StarlarkMethod(
      name = "progress_message",
      doc = "Progress message to set on the proto compiler action.",
      structField = true)
  public abstract String progressMessage();

  @StarlarkMethod(
      name = "mnemonic",
      doc = "Mnemonic to set on the proto compiler action.",
      structField = true)
  public abstract String mnemonic();

  public static ProtoLangToolchainProvider create(
      String outReplacementFormatFlag,
      String pluginFormatFlag,
      FilesToRunProvider pluginExecutable,
      TransitiveInfoCollection runtime,
      ImmutableList<ProtoSource> providedProtoSources,
      FilesToRunProvider protoc,
      ImmutableList<String> protocOpts,
      String progressMessage,
      String mnemonic) {
    return new AutoValue_ProtoLangToolchainProvider(
        outReplacementFormatFlag,
        pluginFormatFlag,
        pluginExecutable,
        runtime,
        providedProtoSources,
        protoc,
        protocOpts,
        progressMessage,
        mnemonic);
  }
}
