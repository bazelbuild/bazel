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
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.starlark.annotations.StarlarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.starlarkbuildapi.ProtoConfigurationApi;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.List;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** Configuration for Protocol Buffer Libraries. */
@Immutable
// This module needs to be exported to Starlark so it can be passed as a mandatory exec/target
// configuration fragment in aspect definitions.
@RequiresOptions(options = {ProtoConfiguration.Options.class})
public class ProtoConfiguration extends Fragment implements ProtoConfigurationApi {

  /** Command line options. */
  public static class Options extends FragmentOptions {

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
        defaultValue = ProtoConstants.DEFAULT_PROTOC_LABEL,
        converter = CoreOptionConverters.LabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "The label of the proto-compiler.")
    public Label protoCompiler;

    @Option(
        name = "proto_toolchain_for_javalite",
        defaultValue = ProtoConstants.DEFAULT_JAVA_LITE_PROTO_LABEL,
        converter = CoreOptionConverters.LabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Label of proto_lang_toolchain() which describes how to compile JavaLite protos")
    public Label protoToolchainForJavaLite;

    @Option(
        name = "proto_toolchain_for_java",
        defaultValue = ProtoConstants.DEFAULT_JAVA_PROTO_LABEL,
        converter = CoreOptionConverters.EmptyToNullLabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Label of proto_lang_toolchain() which describes how to compile Java protos")
    public Label protoToolchainForJava;

    @Option(
        name = "proto_toolchain_for_j2objc",
        defaultValue = ProtoConstants.DEFAULT_J2OBJC_PROTO_LABEL,
        category = "flags",
        converter = CoreOptionConverters.EmptyToNullLabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Label of proto_lang_toolchain() which describes how to compile j2objc protos")
    public Label protoToolchainForJ2objc;

    @Option(
        name = "proto_toolchain_for_cc",
        defaultValue = ProtoConstants.DEFAULT_CC_PROTO_LABEL,
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
        name = "strict_public_imports",
        defaultValue = "off",
        converter = CoreOptionConverters.StrictDepsConverter.class,
        documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
        effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.EAGERNESS_TO_EXIT},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
        help =
            "Unless OFF, checks that a proto_library target explicitly declares all targets used "
                + "in 'import public' as exported.")
    public StrictDepsMode strictPublicImports;

    @Option(
        name = "cc_proto_library_header_suffixes",
        defaultValue = ".pb.h",
        documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Sets the prefixes of header files that a cc_proto_library creates.",
        converter = Converters.CommaSeparatedOptionSetConverter.class)
    public List<String> ccProtoLibraryHeaderSuffixes;

    @Option(
        name = "cc_proto_library_source_suffixes",
        defaultValue = ".pb.cc",
        documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Sets the prefixes of source files that a cc_proto_library creates.",
        converter = Converters.CommaSeparatedOptionSetConverter.class)
    public List<String> ccProtoLibrarySourceSuffixes;

    @Override
    public FragmentOptions getExec() {
      Options exec = (Options) super.getExec();
      exec.protoCompiler = protoCompiler;
      exec.protocOpts = protocOpts;
      exec.experimentalProtoDescriptorSetsIncludeSourceInfo =
          experimentalProtoDescriptorSetsIncludeSourceInfo;
      exec.experimentalProtoExtraActions = experimentalProtoExtraActions;
      exec.protoToolchainForJava = protoToolchainForJava;
      exec.protoToolchainForJ2objc = protoToolchainForJ2objc;
      exec.protoToolchainForJavaLite = protoToolchainForJavaLite;
      exec.protoToolchainForCc = protoToolchainForCc;
      exec.strictProtoDeps = strictProtoDeps;
      exec.strictPublicImports = strictPublicImports;
      exec.ccProtoLibraryHeaderSuffixes = ccProtoLibraryHeaderSuffixes;
      exec.ccProtoLibrarySourceSuffixes = ccProtoLibrarySourceSuffixes;
      return exec;
    }
  }

  private final ImmutableList<String> protocOpts;
  private final ImmutableList<String> ccProtoLibraryHeaderSuffixes;
  private final ImmutableList<String> ccProtoLibrarySourceSuffixes;
  private final Options options;

  public ProtoConfiguration(BuildOptions buildOptions) {
    Options options = buildOptions.get(Options.class);
    this.protocOpts = ImmutableList.copyOf(options.protocOpts);
    this.ccProtoLibraryHeaderSuffixes = ImmutableList.copyOf(options.ccProtoLibraryHeaderSuffixes);
    this.ccProtoLibrarySourceSuffixes = ImmutableList.copyOf(options.ccProtoLibrarySourceSuffixes);
    this.options = options;
  }

  @StarlarkMethod(name = "experimental_protoc_opts", structField = true, documented = false)
  public ImmutableList<String> protocOptsForStarlark() throws EvalException {
    return protocOpts();
  }

  public ImmutableList<String> protocOpts() {
    return protocOpts;
  }

  @StarlarkMethod(
      name = "experimental_proto_descriptorsets_include_source_info",
      useStarlarkThread = true,
      documented = false)
  public boolean experimentalProtoDescriptorSetsIncludeSourceInfoForStarlark(StarlarkThread thread)
      throws EvalException {
    BuiltinRestriction.failIfCalledOutsideBuiltins(thread);
    return experimentalProtoDescriptorSetsIncludeSourceInfo();
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

  @StarlarkConfigurationField(
      name = "proto_compiler",
      doc = "Label for the proto compiler.",
      defaultLabel = ProtoConstants.DEFAULT_PROTOC_LABEL)
  public Label protoCompiler() {
    return options.protoCompiler;
  }

  @StarlarkConfigurationField(
      name = "proto_toolchain_for_java",
      doc = "Label for the java proto toolchains.",
      defaultLabel = ProtoConstants.DEFAULT_JAVA_PROTO_LABEL)
  public Label protoToolchainForJava() {
    return options.protoToolchainForJava;
  }

  @StarlarkConfigurationField(
      name = "proto_toolchain_for_j2objc",
      doc = "Label for the j2objc toolchains.",
      defaultLabel = ProtoConstants.DEFAULT_J2OBJC_PROTO_LABEL)
  public Label protoToolchainForJ2objc() {
    return options.protoToolchainForJ2objc;
  }

  @StarlarkConfigurationField(
      name = "proto_toolchain_for_java_lite",
      doc = "Label for the java lite proto toolchains.",
      defaultLabel = ProtoConstants.DEFAULT_JAVA_LITE_PROTO_LABEL)
  public Label protoToolchainForJavaLite() {
    return options.protoToolchainForJavaLite;
  }

  @StarlarkConfigurationField(
      name = "proto_toolchain_for_cc",
      doc = "Label for the cc proto toolchains.",
      defaultLabel = ProtoConstants.DEFAULT_CC_PROTO_LABEL)
  public Label protoToolchainForCc() {
    return options.protoToolchainForCc;
  }

  @StarlarkMethod(name = "strict_proto_deps", useStarlarkThread = true, documented = false)
  public String strictProtoDepsForStarlark(StarlarkThread thread) throws EvalException {
    BuiltinRestriction.failIfCalledOutsideBuiltins(thread);
    return strictProtoDeps().toString();
  }

  @StarlarkMethod(name = "strict_public_imports", useStarlarkThread = true, documented = false)
  public String strictPublicImportsForStarlark(StarlarkThread thread) throws EvalException {
    BuiltinRestriction.failIfCalledOutsideBuiltins(thread);
    return options.strictPublicImports.toString();
  }

  public StrictDepsMode strictProtoDeps() {
    return options.strictProtoDeps;
  }

  @StarlarkMethod(
      name = "cc_proto_library_header_suffixes",
      useStarlarkThread = true,
      documented = false)
  public List<String> ccProtoLibraryHeaderSuffixesForStarlark(StarlarkThread thread)
      throws EvalException {
    BuiltinRestriction.failIfCalledOutsideBuiltins(thread);
    return ccProtoLibraryHeaderSuffixes();
  }

  public List<String> ccProtoLibraryHeaderSuffixes() {
    return ccProtoLibraryHeaderSuffixes;
  }

  @StarlarkMethod(
      name = "cc_proto_library_source_suffixes",
      useStarlarkThread = true,
      documented = false)
  public List<String> ccProtoLibrarySourceSuffixesForStarlark(StarlarkThread thread)
      throws EvalException {
    BuiltinRestriction.failIfCalledOutsideBuiltins(thread);
    return ccProtoLibrarySourceSuffixes();
  }

  public List<String> ccProtoLibrarySourceSuffixes() {
    return ccProtoLibrarySourceSuffixes;
  }
}
