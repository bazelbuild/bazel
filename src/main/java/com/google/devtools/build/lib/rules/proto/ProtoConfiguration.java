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
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
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
    @Option(
      name = "protocopt",
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Additional options to pass to the protobuf compiler."
    )
    public List<String> protocOpts;

    @Option(
      name = "experimental_proto_extra_actions",
      defaultValue = "false",
      category = "experimental",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "Run extra actions for alternative Java api versions in a proto_library."
    )
    public boolean experimentalProtoExtraActions;

    @Option(
      name = "proto_compiler",
      defaultValue = "@com_google_protobuf//:protoc",
      category = "flags",
      converter = BuildConfiguration.LabelConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "The label of the proto-compiler."
    )
    public Label protoCompiler;

    @Option(
      name = "proto_toolchain_for_javalite",
      defaultValue = "@com_google_protobuf_javalite//:javalite_toolchain",
      category = "flags",
      converter = BuildConfiguration.EmptyToNullLabelConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Label of proto_lang_toolchain() which describes how to compile JavaLite protos"
    )
    public Label protoToolchainForJavaLite;

    @Option(
      name = "proto_toolchain_for_java",
      defaultValue = "@com_google_protobuf_java//:java_toolchain",
      category = "flags",
      converter = BuildConfiguration.EmptyToNullLabelConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Label of proto_lang_toolchain() which describes how to compile Java protos"
    )
    public Label protoToolchainForJava;

    @Option(
      name = "proto_toolchain_for_cc",
      defaultValue = "@com_google_protobuf_cc//:cc_toolchain",
      category = "flags",
      converter = BuildConfiguration.EmptyToNullLabelConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Label of proto_lang_toolchain() which describes how to compile C++ protos"
    )
    public Label protoToolchainForCc;

    @Option(
      name = "strict_proto_deps",
      defaultValue = "strict",
      converter = BuildConfiguration.StrictDepsConverter.class,
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If true, checks that a proto_library target explicitly declares all directly "
              + "used targets as dependencies."
    )
    public StrictDepsMode strictProtoDeps;

    @Option(
      name = "cc_proto_library_header_suffixes",
      defaultValue = ".pb.h",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Sets the prefixes of header files that a cc_proto_library creates.",
      converter = Converters.CommaSeparatedOptionListConverter.class
    )
    public List<String> ccProtoLibraryHeaderSuffixes;

    @Option(
      name = "cc_proto_library_source_suffixes",
      defaultValue = ".pb.cc",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Sets the prefixes of source files that a cc_proto_library creates.",
      converter = Converters.CommaSeparatedOptionListConverter.class
    )
    public List<String> ccProtoLibrarySourceSuffixes;

    // TODO(b/64032754): Remove once there's no 'correctRollupTransitiveProtoRuntimes' in the global
    //     blazerc.
    @Option(
      name = "correctRollupTransitiveProtoRuntimes",
      defaultValue = "true",
      category = "rollout",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "ignored"
    )
    public boolean correctRollupTransitiveProtoRuntimes;

    // TODO(b/62710272): Remove once there's no 'jplNonStrictDepsLikePl' in the global blazerc.
    @Option(
      name = "jplNonStrictDepsLikePl",
      defaultValue = "true",
      category = "rollout",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "ignored"
    )
    public boolean jplNonStrictDepsLikePl;

    @Override
    public FragmentOptions getHost(boolean fallback) {
      Options host = (Options) super.getHost(fallback);
      host.protoCompiler = protoCompiler;
      host.protocOpts = protocOpts;
      host.experimentalProtoExtraActions = experimentalProtoExtraActions;
      host.protoCompiler = protoCompiler;
      host.protoToolchainForJava = protoToolchainForJava;
      host.protoToolchainForJavaLite = protoToolchainForJavaLite;
      host.protoToolchainForCc = protoToolchainForCc;
      host.strictProtoDeps = strictProtoDeps;
      host.ccProtoLibraryHeaderSuffixes = ccProtoLibraryHeaderSuffixes;
      host.ccProtoLibrarySourceSuffixes = ccProtoLibrarySourceSuffixes;
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

  private final ImmutableList<String> protocOpts;
  private final ImmutableList<String> ccProtoLibraryHeaderSuffixes;
  private final ImmutableList<String> ccProtoLibrarySourceSuffixes;
  private final Options options;

  public ProtoConfiguration(Options options) {
    this.protocOpts = ImmutableList.copyOf(options.protocOpts);
    this.ccProtoLibraryHeaderSuffixes = ImmutableList.copyOf(options.ccProtoLibraryHeaderSuffixes);
    this.ccProtoLibrarySourceSuffixes = ImmutableList.copyOf(options.ccProtoLibrarySourceSuffixes);
    this.options = options;
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
    return options.experimentalProtoExtraActions;
  }

  public Label protoCompiler() {
    return options.protoCompiler;
  }

  public Label protoToolchainForJava() {
    return options.protoToolchainForJava;
  }

  public Label protoToolchainForJavaLite() {
    return options.protoToolchainForJavaLite;
  }

  public Label protoToolchainForCc() {
    return options.protoToolchainForCc;
  }

  public StrictDepsMode strictProtoDeps() {
    return options.strictProtoDeps;
  }

  public List<String> ccProtoLibraryHeaderSuffixes() {
    return ccProtoLibraryHeaderSuffixes;
  }

  public List<String> ccProtoLibrarySourceSuffixes() {
    return ccProtoLibrarySourceSuffixes;
  }

}
