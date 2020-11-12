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
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.ProtoConfigurationApi;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.List;

/** Configuration for Protocol Buffer Libraries. */
@Immutable
// This module needs to be exported to Starlark so it can be passed as a mandatory host/target
// configuration fragment in aspect definitions.
@RequiresOptions(options = {ProtoConfiguration.Options.class})
public class ProtoConfiguration extends Fragment implements ProtoConfigurationApi {
  /** Command line options. */
  public static class Options extends FragmentOptions {
    @Option(
        name = "incompatible_generated_protos_in_virtual_imports",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help =
            "If set, generated .proto files are put into a virtual import directory. For more "
                + "information, see https://github.com/bazelbuild/bazel/issues/9215")
    public boolean generatedProtosInVirtualImports;

    @Option(
        name = "protocopt",
        allowMultiple = true,
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        help = "Additional options to pass to the protobuf compiler.")
    public List<String> protocOpts;

    @Option(
      name = "experimental_proto_extra_actions",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "Run extra actions for alternative Java api versions in a proto_library."
    )
    public boolean experimentalProtoExtraActions;

    @Option(
        name = "experimental_proto_descriptor_sets_include_source_info",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL},
        help = "Run extra actions for alternative Java api versions in a proto_library.")
    public boolean experimentalProtoDescriptorSetsIncludeSourceInfo;

    @Option(
        name = "proto_compiler",
        defaultValue = "@com_google_protobuf//:protoc",
        converter = CoreOptionConverters.LabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "The label of the proto-compiler.")
    public Label protoCompiler;

    @Option(
        name = "proto_toolchain_for_javalite",
        defaultValue = "@com_google_protobuf//:javalite_toolchain",
        converter = CoreOptionConverters.LabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Label of proto_lang_toolchain() which describes how to compile JavaLite protos")
    public Label protoToolchainForJavaLite;

    @Option(
        name = "proto_toolchain_for_java",
        defaultValue = "@com_google_protobuf//:java_toolchain",
        converter = CoreOptionConverters.EmptyToNullLabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Label of proto_lang_toolchain() which describes how to compile Java protos")
    public Label protoToolchainForJava;

    @Option(
        name = "proto_toolchain_for_j2objc",
        defaultValue = "@bazel_tools//tools/j2objc:j2objc_proto_toolchain",
        category = "flags",
        converter = CoreOptionConverters.EmptyToNullLabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Label of proto_lang_toolchain() which describes how to compile j2objc protos")
    public Label protoToolchainForJ2objc;

    @Option(
        name = "proto_toolchain_for_cc",
        defaultValue = "@com_google_protobuf//:cc_toolchain",
        converter = CoreOptionConverters.EmptyToNullLabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Label of proto_lang_toolchain() which describes how to compile C++ protos")
    public Label protoToolchainForCc;

    @Option(
        name = "strict_proto_deps",
        defaultValue = "error",
        converter = CoreOptionConverters.StrictDepsConverter.class,
        documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
        effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.EAGERNESS_TO_EXIT},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
        help =
            "Unless OFF, checks that a proto_library target explicitly declares all directly "
                + "used targets as dependencies.")
    public StrictDepsMode strictProtoDeps;

    @Option(
      name = "cc_proto_library_header_suffixes",
      defaultValue = ".pb.h",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Sets the prefixes of header files that a cc_proto_library creates.",
      converter = Converters.CommaSeparatedOptionListConverter.class
    )
    public List<String> ccProtoLibraryHeaderSuffixes;

    @Option(
      name = "cc_proto_library_source_suffixes",
      defaultValue = ".pb.cc",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Sets the prefixes of source files that a cc_proto_library creates.",
      converter = Converters.CommaSeparatedOptionListConverter.class
    )
    public List<String> ccProtoLibrarySourceSuffixes;

    @Option(
        name = "experimental_java_proto_add_allowed_public_imports",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL},
        help = "If true, add --allowed_public_imports to the java compile actions.")
    public boolean experimentalJavaProtoAddAllowedPublicImports;

    @Override
    public FragmentOptions getHost() {
      Options host = (Options) super.getHost();
      host.protoCompiler = protoCompiler;
      host.protocOpts = protocOpts;
      host.experimentalProtoDescriptorSetsIncludeSourceInfo =
          experimentalProtoDescriptorSetsIncludeSourceInfo;
      host.experimentalProtoExtraActions = experimentalProtoExtraActions;
      host.protoCompiler = protoCompiler;
      host.protoToolchainForJava = protoToolchainForJava;
      host.protoToolchainForJ2objc = protoToolchainForJ2objc;
      host.protoToolchainForJavaLite = protoToolchainForJavaLite;
      host.protoToolchainForCc = protoToolchainForCc;
      host.strictProtoDeps = strictProtoDeps;
      host.ccProtoLibraryHeaderSuffixes = ccProtoLibraryHeaderSuffixes;
      host.ccProtoLibrarySourceSuffixes = ccProtoLibrarySourceSuffixes;
      host.experimentalJavaProtoAddAllowedPublicImports =
          experimentalJavaProtoAddAllowedPublicImports;
      host.generatedProtosInVirtualImports = generatedProtosInVirtualImports;
      return host;
    }
  }

  /**
   * Loader class for proto.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new ProtoConfiguration(buildOptions);
    }

    @Override
    public Class<? extends Fragment> creates() {
      return ProtoConfiguration.class;
    }
  }

  private final ImmutableList<String> protocOpts;
  private final ImmutableList<String> ccProtoLibraryHeaderSuffixes;
  private final ImmutableList<String> ccProtoLibrarySourceSuffixes;
  private final Options options;

  private ProtoConfiguration(BuildOptions buildOptions) {
    Options options = buildOptions.get(Options.class);
    this.protocOpts = ImmutableList.copyOf(options.protocOpts);
    this.ccProtoLibraryHeaderSuffixes = ImmutableList.copyOf(options.ccProtoLibraryHeaderSuffixes);
    this.ccProtoLibrarySourceSuffixes = ImmutableList.copyOf(options.ccProtoLibrarySourceSuffixes);
    this.options = options;
  }

  public ImmutableList<String> protocOpts() {
    return protocOpts;
  }

  public boolean experimentalProtoDescriptorSetsIncludeSourceInfo() {
    return options.experimentalProtoDescriptorSetsIncludeSourceInfo;
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

  public Label protoToolchainForJ2objc() {
    return options.protoToolchainForJ2objc;
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

  public boolean strictPublicImports() {
    return options.experimentalJavaProtoAddAllowedPublicImports;
  }

  public boolean generatedProtosInVirtualImports() {
    return options.generatedProtosInVirtualImports;
  }
}
