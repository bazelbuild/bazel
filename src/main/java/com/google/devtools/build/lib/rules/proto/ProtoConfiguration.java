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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.common.options.Option;
import java.util.List;

/**
 * Configuration for Protocol Buffer Libraries.
 */
@Immutable
// This module needs to be exported to Skylark so it can be passed as a mandatory host/target
// configuration fragment in aspect definitions.
@SkylarkModule(
    name = "proto",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT,
    doc = "A configuration fragment representing protocol buffers."
)
public class ProtoConfiguration extends Fragment {

  /**
   * Command line options.
   */
  public static class Options extends FragmentOptions {
    @Option(name = "protocopt",
        allowMultiple = true,
        defaultValue = "",
        category = "flags",
        help = "Additional options to pass to the protobuf compiler.")
    public List<String> protocOpts;

    @Option(
      name = "experimental_proto_extra_actions",
      defaultValue = "false",
      category = "experimental",
      help = "Run extra actions for alternative Java api versions in a proto_library."
    )
    public boolean experimentalProtoExtraActions;

    @Option(
      name = "proto_compiler",
      defaultValue = "null",
      category = "version",
      converter = BuildConfiguration.LabelConverter.class,
      help = "The label of the proto-compiler."
    )
    public Label protoCompiler;

    // TODO(b/31775048): Replace with a toolchain
    /** This is experimental, and is subject to change without warning. */
    @Option(
      name = "proto_compiler_java_flags",
      defaultValue = "--java_out=shared,immutable:%s",
      category = "experimental",
      help = "The flags to pass to proto-compiler when generating Java protos."
    )
    public String protoCompilerJavaFlags;

    // TODO(b/31775048): Replace with a toolchain
    /** This is experimental, and is subject to change without warning. */
    @Option(
      name = "proto_compiler_java_blacklisted_protos",
      defaultValue = "",
      category = "experimental",
      converter = BuildConfiguration.LabelListConverter.class,
      help = "A label of a filegroup of .proto files that we shouldn't generate sources for."
    )
    public List<Label> protoCompilerJavaBlacklistedProtos;

    // TODO(b/31775048): Replace with a toolchain
    /** This is experimental, and is subject to change without warning. */
    @Option(
      name = "proto_compiler_javalite_flags",
      defaultValue = "--javalite_out=%s",
      category = "experimental",
      help = "The flags to pass to proto-compiler when generating JavaLite protos."
    )
    public String protoCompilerJavaLiteFlags;

    // TODO(b/31775048): Replace with a toolchain
    /** This is experimental, and is subject to change without warning. */
    @Option(
      name = "proto_compiler_javalite_plugin",
      defaultValue = "",
      category = "experimental",
      converter = BuildConfiguration.EmptyToNullLabelConverter.class,
      help = "A label for the javalite proto-compiler plugin, if needed."
    )
    public Label protoCompilerJavaLitePlugin;

    @Option(
      name = "proto_toolchain_for_javalite",
      defaultValue = "@com_google_protobuf_javalite//:javalite_toolchain",
      category = "flags",
      converter = BuildConfiguration.EmptyToNullLabelConverter.class,
      help = "Label of proto_lang_toolchain() which describes how to compile JavaLite protos"
    )
    public Label protoToolchainForJavaLite;

    @Option(
      name = "proto_toolchain_for_java",
      defaultValue = "@com_google_protobuf_java//:java_toolchain",
      category = "flags",
      converter = BuildConfiguration.EmptyToNullLabelConverter.class,
      help = "Label of proto_lang_toolchain() which describes how to compile Java protos"
    )
    public Label protoToolchainForJava;

    @Option(
      name = "use_toolchain_for_java_proto",
      defaultValue = "false",
      category = "experimental",
      help =
          "If true, --proto_toolchain_for_java will be used for java_proto_library. "
              + "This flag is an escape-hatch and should be removed once toolchain-based builds "
              + "are tested."
    )
    public boolean useToolchainForJavaProto;

    @Option(
      name = "strict_proto_deps",
      defaultValue = "error",
      converter = BuildConfiguration.StrictDepsConverter.class,
      category = "semantics",
      help =
          "If true, checks that a proto_library target explicitly declares all directly "
              + "used targets as dependencies."
    )
    public StrictDepsMode strictProtoDeps;

    @Option(
      name = "output_descriptor_set",
      defaultValue = "true",
      category = "experimental",
      help = "When true, a proto_library will produce a descriptor set proto in its outputs."
    )
    public boolean outputDescriptorSet;

    @Override
    public FragmentOptions getHost(boolean fallback) {
      Options host = (Options) super.getHost(fallback);
      host.protoCompiler = protoCompiler;
      host.protocOpts = protocOpts;
      host.experimentalProtoExtraActions = experimentalProtoExtraActions;
      host.protoCompiler = protoCompiler;
      host.protoCompilerJavaFlags = protoCompilerJavaFlags;
      host.protoCompilerJavaBlacklistedProtos = protoCompilerJavaBlacklistedProtos;
      host.protoCompilerJavaLiteFlags = protoCompilerJavaLiteFlags;
      host.protoCompilerJavaLitePlugin = protoCompilerJavaLitePlugin;
      host.protoToolchainForJava = protoToolchainForJava;
      host.protoToolchainForJavaLite = protoToolchainForJavaLite;
      host.useToolchainForJavaProto = useToolchainForJavaProto;
      host.strictProtoDeps = strictProtoDeps;
      host.outputDescriptorSet = outputDescriptorSet;
      return host;
    }
  }

  /**
   * Loader class for proto.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new ProtoConfiguration(buildOptions.get(Options.class));
    }

    @Override
    public Class<? extends Fragment> creates() {
      return ProtoConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(Options.class);
    }
  }

  private final boolean experimentalProtoExtraActions;
  private final ImmutableList<String> protocOpts;
  private final Label protoCompiler;
  private final String protoCompilerJavaFlags;
  private final List<Label> protoCompilerJavaBlacklistedProtos;
  private final String protoCompilerJavaLiteFlags;
  private final Label protoCompilerJavaLitePlugin;
  private final Label protoToolchainForJava;
  private final Label protoToolchainForJavaLite;
  private final boolean useToolchainForJavaProto;
  private final StrictDepsMode strictProtoDeps;
  private final boolean outputDescriptorSet;

  public ProtoConfiguration(Options options) {
    this.experimentalProtoExtraActions = options.experimentalProtoExtraActions;
    this.protocOpts = ImmutableList.copyOf(options.protocOpts);
    this.protoCompiler = options.protoCompiler;
    this.protoCompilerJavaFlags = options.protoCompilerJavaFlags;
    this.protoCompilerJavaLiteFlags = options.protoCompilerJavaLiteFlags;
    this.protoCompilerJavaLitePlugin = options.protoCompilerJavaLitePlugin;
    this.protoCompilerJavaBlacklistedProtos = options.protoCompilerJavaBlacklistedProtos;
    this.protoToolchainForJava = options.protoToolchainForJava;
    this.protoToolchainForJavaLite = options.protoToolchainForJavaLite;
    this.useToolchainForJavaProto = options.useToolchainForJavaProto;
    this.strictProtoDeps = options.strictProtoDeps;
    this.outputDescriptorSet = options.outputDescriptorSet;
  }

  public ImmutableList<String> protocOpts() {
    return protocOpts;
  }

  /**
   * Returns true if we will run extra actions for actions that are not run by default. If this
   * is enabled, e.g. all extra_actions for alternative api-versions or language-flavours of a
   * proto_library target are run.
   */
  public boolean runExperimentalProtoExtraActions() {
    return experimentalProtoExtraActions;
  }

  public Label protoCompiler() {
    return protoCompiler;
  }

  public String protoCompilerJavaFlags() {
    return protoCompilerJavaFlags;
  }

  public String protoCompilerJavaLiteFlags() {
    return protoCompilerJavaLiteFlags;
  }

  public Label protoCompilerJavaLitePlugin() {
    return protoCompilerJavaLitePlugin;
  }

  public List<Label> protoCompilerJavaBlacklistedProtos() {
    return protoCompilerJavaBlacklistedProtos;
  }

  public Label protoToolchainForJava() {
    return protoToolchainForJava;
  }

  public Label protoToolchainForJavaLite() {
    return protoToolchainForJavaLite;
  }

  public boolean useToolchainForJavaProto() {
    return useToolchainForJavaProto;
  }

  public StrictDepsMode strictProtoDeps() {
    return strictProtoDeps;
  }

  public boolean outputDescriptorSet() {
    return outputDescriptorSet;
  }
}
