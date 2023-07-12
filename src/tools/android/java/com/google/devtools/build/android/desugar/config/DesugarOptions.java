/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.google.devtools.build.android.desugar.config;

import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.List;

/** Commandline options for {@link com.google.devtools.build.android.desugar.Desugar}. */
public class DesugarOptions extends OptionsBase {

  @Option(
      name = "input",
      allowMultiple = true,
      defaultValue = "null",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      abbrev = 'i',
      help =
          "Input Jar or directory with classes to desugar (required, the n-th input is paired"
              + " with the n-th output).")
  public List<Path> inputJars;

  @Option(
      name = "classpath_entry",
      allowMultiple = true,
      defaultValue = "null",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      help =
          "Ordered classpath (Jar or directory) to resolve symbols in the --input Jar, like "
              + "javac's -cp flag.")
  public List<Path> classpath;

  @Option(
      name = "bootclasspath_entry",
      allowMultiple = true,
      defaultValue = "null",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      help =
          "Bootclasspath that was used to compile the --input Jar with, like javac's "
              + "-bootclasspath flag (required).")
  public List<Path> bootclasspath;

  @Option(
      name = "allow_empty_bootclasspath",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN})
  public boolean allowEmptyBootclasspath;

  @Option(
      name = "only_desugar_javac9_for_lint",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "A temporary flag specifically for android lint, subject to removal anytime (DO NOT"
              + " USE)")
  public boolean onlyDesugarJavac9ForLint;

  @Option(
      name = "rewrite_calls_to_long_compare",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Rewrite calls to Long.compare(long, long) to the JVM instruction lcmp "
              + "regardless of --min_sdk_version.",
      category = "misc")
  public boolean alwaysRewriteLongCompare;

  @Option(
      name = "output",
      allowMultiple = true,
      defaultValue = "null",
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = PathConverter.class,
      abbrev = 'o',
      help =
          "Output Jar or directory to write desugared classes into (required, the n-th output is "
              + "paired with the n-th input, output must be a Jar if input is a Jar).")
  public List<Path> outputJars;

  @Option(
      name = "verbose",
      defaultValue = "false",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      abbrev = 'v',
      help = "Enables verbose debugging output.")
  public boolean verbose;

  @Option(
      name = "min_sdk_version",
      defaultValue = "1",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Minimum targeted sdk version.  If >= 24, enables default methods in interfaces.")
  public int minSdkVersion;

  @Option(
      name = "emit_dependency_metadata_as_needed",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Whether to emit META-INF/desugar_deps as needed for later consistency checking.")
  public boolean emitDependencyMetadata;

  @Option(
      name = "best_effort_tolerate_missing_deps",
      defaultValue = "true",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Whether to tolerate missing dependencies on the classpath in some cases.  You should "
              + "strive to set this flag to false.")
  public boolean tolerateMissingDependencies;

  @Option(
      name = "desugar_supported_core_libs",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Enable core library desugaring, which requires configuration with related flags.")
  public boolean desugarCoreLibs;

  @Option(
      name = "desugar_interface_method_bodies_if_needed",
      defaultValue = "true",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Rewrites default and static methods in interfaces if --min_sdk_version < 24. This "
              + "only works correctly if subclasses of rewritten interfaces as well as uses of "
              + "static interface methods are run through this tool as well.")
  public boolean desugarInterfaceMethodBodiesIfNeeded;

  @Option(
      name = "desugar_try_with_resources_if_needed",
      defaultValue = "true",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Rewrites try-with-resources statements if --min_sdk_version < 19.")
  public boolean desugarTryWithResourcesIfNeeded;

  @Option(
      name = "desugar_try_with_resources_omit_runtime_classes",
      defaultValue = "false",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Omits the runtime classes necessary to support try-with-resources from the output."
              + " This property has effect only if --desugar_try_with_resources_if_needed is"
              + " used.")
  public boolean desugarTryWithResourcesOmitRuntimeClasses;

  @Option(
      name = "generate_base_classes_for_default_methods",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "If desugaring default methods, generate abstract base classes for them. "
              + "This reduces default method stubs in hand-written subclasses.")
  public boolean generateBaseClassesForDefaultMethods;

  @Option(
      name = "copy_bridges_from_classpath",
      defaultValue = "false",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Copy bridges from classpath to desugared classes.")
  public boolean copyBridgesFromClasspath;

  @Option(
      name = "core_library",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Enables rewriting to desugar java.* classes.")
  public boolean coreLibrary;

  /** Type prefixes that we'll move to a custom package. */
  @Option(
      name = "rewrite_core_library_prefix",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Assume the given java.* prefixes are desugared.")
  public List<String> rewriteCoreLibraryPrefixes;

  /** Interfaces whose default and static interface methods we'll emulate. */
  @Option(
      name = "emulate_core_library_interface",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Assume the given java.* interfaces are emulated.")
  public List<String> emulateCoreLibraryInterfaces;

  /** Members not to rewrite. */
  @Option(
      name = "dont_rewrite_core_library_invocation",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Method invocations not to rewrite, given as \"class/Name#method\".")
  public List<String> dontTouchCoreLibraryMembers;

  @Option(
      name = "auto_desugar_shadowed_api_use",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Enables automatic invocable and overridable desugar-shadowed APIs.")
  public boolean autoDesugarShadowedApiUse;

  /** Set to work around b/62623509 with JaCoCo versions prior to 0.7.9. */
  // TODO(kmb): Remove when Android Studio doesn't need it anymore (see b/37116789)
  @Option(
      name = "legacy_jacoco_fix",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Consider setting this flag if you're using JaCoCo versions prior to 0.7.9 to work"
              + " around issues with coverage instrumentation in default and static interface"
              + " methods. This flag may be removed when no longer needed.")
  public boolean legacyJacocoFix;

  /** Convert Java 11 nest-based access control to bridge-based access control. */
  @Option(
      name = "desugar_nest_based_private_access",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Desugar JVM 11 native supported accessing private nest members with bridge method"
              + " based accessors. This flag includes desugaring private interface methods.")
  public boolean desugarNestBasedPrivateAccess;

  /**
   * Convert Java 9 invokedynamic-based string concatenations to StringBuilder-based
   * concatenations. @see https://openjdk.java.net/jeps/280
   */
  @Option(
      name = "desugar_indy_string_concat",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Desugar JVM 9 string concatenation operations to string builder based"
              + " implementations.")
  public boolean desugarIndifyStringConcat;

  public static DesugarOptions parseCommandLineOptions(String[] args) {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(DesugarOptions.class)
            .allowResidue(false)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    parser.parseAndExitUponError(args);
    DesugarOptions options = parser.getOptions(DesugarOptions.class);

    return options;
  }
}
