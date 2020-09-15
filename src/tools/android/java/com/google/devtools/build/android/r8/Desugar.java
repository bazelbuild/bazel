// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8;

import static com.google.common.base.Preconditions.checkArgument;

import com.android.tools.r8.ArchiveClassFileProvider;
import com.android.tools.r8.ClassFileResourceProvider;
import com.android.tools.r8.CompilationFailedException;
import com.android.tools.r8.D8;
import com.android.tools.r8.D8Command;
import com.android.tools.r8.Diagnostic;
import com.android.tools.r8.DiagnosticsHandler;
import com.android.tools.r8.StringResource;
import com.android.tools.r8.errors.InterfaceDesugarMissingTypeDiagnostic;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.desugar.DependencyCollector;
import com.google.devtools.build.android.r8.desugar.OrderedClassFileResourceProvider;
import com.google.devtools.build.android.r8.desugar.OutputConsumer;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Desugar compatible wrapper based on D8 desugaring engine */
public class Desugar {

  public static final String DESUGAR_DEPS_FILENAME = "META-INF/desugar_deps";

  /** Commandline options for {@link com.google.devtools.build.android.r8.Desugar}. */
  public static class DesugarOptions extends OptionsBase {

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

    /** Members that we will retarget to the given new owner. */
    @Option(
        name = "retarget_core_library_member",
        defaultValue = "null",
        allowMultiple = true,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Method invocations to retarget, given as \"class/Name#member->new/class/Name\".  "
                + "The new owner is blindly assumed to exist.")
    public List<String> retargetCoreLibraryMembers;

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
        name = "preserve_core_library_override",
        defaultValue = "null",
        allowMultiple = true,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Core library methods given as \"class/Name#method\" whose overrides should be"
                + " preserved.  Typically this is useful when the given class itself isn't"
                + " desugared.")
    public List<String> preserveCoreLibraryOverrides;

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

    @Option(
        name = "persistent_worker",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {OptionMetadataTag.HIDDEN},
        help = "Run as a Bazel persistent worker.")
    public boolean persistentWorker;

    @Option(
        name = "desugared_lib_config",
        defaultValue = "null",
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = ExistingPathConverter.class,
        help =
            "Specify desugared library configuration. "
                + "The input file is a desugared library configuration (json)")
    public Path desugaredLibConfig;
  }

  private final DesugarOptions options;

  private Desugar(DesugarOptions options) {
    this.options = options;
  }

  private static DesugarOptions parseCommandLineOptions(String[] args) {
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

  /**
   * Returns a dependency collector for use with a single input Jar. If {@link
   * DesugarOptions#emitDependencyMetadata} is set, this method instantiates the collector
   * reflectively to allow compiling and using the desugar tool without this mechanism.
   */
  private DependencyCollector createDependencyCollector() {
    if (options.emitDependencyMetadata) {
      try {
        return (DependencyCollector)
            Thread.currentThread()
                .getContextClassLoader()
                .loadClass(
                    "com.google.devtools.build.android.desugar.dependencies.MetadataCollector")
                .getConstructor(Boolean.TYPE)
                .newInstance(options.tolerateMissingDependencies);
      } catch (ReflectiveOperationException | SecurityException e) {
        throw new IllegalStateException("Can't emit desugaring metadata as requested", e);
      }
    } else if (options.tolerateMissingDependencies) {
      return DependencyCollector.NoWriteCollectors.NOOP;
    } else {
      return DependencyCollector.NoWriteCollectors.FAIL_ON_MISSING;
    }
  }

  private class DesugarDiagnosticsHandler implements DiagnosticsHandler {

    OutputConsumer outputConsumer;

    private DesugarDiagnosticsHandler(OutputConsumer outputConsumer) {
      this.outputConsumer = outputConsumer;
    }

    @Override
    public void warning(Diagnostic warning) {
      if (warning instanceof InterfaceDesugarMissingTypeDiagnostic) {
        InterfaceDesugarMissingTypeDiagnostic missingTypeDiagnostic =
            (InterfaceDesugarMissingTypeDiagnostic) warning;
        outputConsumer.missingImplementedInterface(
            DescriptorUtils.descriptorToBinaryName(
                missingTypeDiagnostic.getContextType().getDescriptor()),
            DescriptorUtils.descriptorToBinaryName(
                missingTypeDiagnostic.getMissingType().getDescriptor()));
      }
      DiagnosticsHandler.super.warning(warning);
    }
  }

  private void desugar(
      List<ClassFileResourceProvider> bootclasspathProviders,
      ClassFileResourceProvider classpath,
      Path input,
      Path output,
      Path desugaredLibConfig)
      throws CompilationFailedException {
    checkArgument(!Files.isDirectory(input), "Input must be a jar (%s is a directory)", input);
    DependencyCollector dependencyCollector = createDependencyCollector();
    OutputConsumer consumer = new OutputConsumer(output, dependencyCollector);
    D8Command.Builder builder =
        D8Command.builder(new DesugarDiagnosticsHandler(consumer))
            .addClasspathResourceProvider(classpath)
            .addProgramFiles(input)
            .setIntermediate(true)
            .setMinApiLevel(options.minSdkVersion)
            .setProgramConsumer(consumer);
    bootclasspathProviders.forEach(builder::addLibraryResourceProvider);
    if (desugaredLibConfig != null) {
      builder.addDesugaredLibraryConfiguration(StringResource.fromFile(desugaredLibConfig));
    }
    D8.run(builder.build());
  }

  private void desugar() throws CompilationFailedException, IOException {
    // Prepare bootclasspath and classpath. Some jars on the classpath are considered to be
    // bootclasspath, and are moved there.
    ImmutableList.Builder<ClassFileResourceProvider> bootclasspathProvidersBuilder =
        ImmutableList.builder();
    for (Path path : options.bootclasspath) {
      bootclasspathProvidersBuilder.add(new ArchiveClassFileProvider(path));
    }
    ImmutableList.Builder<ClassFileResourceProvider> classpathProvidersBuilder =
        ImmutableList.builder();
    for (Path path : options.classpath) {
      ClassFileResourceProvider provider = new ArchiveClassFileProvider(path);
      if (isPlatform(provider)) {
        bootclasspathProvidersBuilder.add(provider);
      } else {
        classpathProvidersBuilder.add(provider);
      }
    }

    ImmutableList<ClassFileResourceProvider> bootclasspathProviders =
        bootclasspathProvidersBuilder.build();
    OrderedClassFileResourceProvider classpathProvider =
        new OrderedClassFileResourceProvider(
            bootclasspathProviders, classpathProvidersBuilder.build());

    // Desugar the input jars into the specified output jars.
    for (int i = 0; i < options.inputJars.size(); i++) {
      desugar(
          bootclasspathProviders,
          classpathProvider,
          options.inputJars.get(i),
          options.outputJars.get(i),
          options.desugaredLibConfig);
    }
  }

  private static boolean isPlatform(ClassFileResourceProvider provider) {
    // See b/153106333.
    boolean mightBePlatform = false;
    for (String descriptor : provider.getClassDescriptors()) {
      // If the jar contains classes in the package android.car.content this could be a platform
      // library. However, if it also has classes in the package android.car.test it is not.
      if (!mightBePlatform && descriptor.startsWith("Landroid/car/content/")) {
        mightBePlatform = true;
      }
      if (descriptor.startsWith("Landroid/car/test/")) {
        return false;
      }
    }
    // Found classes in the package android.car.content and not in the package android.car.test.
    return mightBePlatform;
  }

  private static void validateOptions(DesugarOptions options) {
    if (options.allowEmptyBootclasspath) {
      throw new AssertionError("--allow_empty_bootclasspath is not supported");
    }
    if (options.onlyDesugarJavac9ForLint) {
      throw new AssertionError("--only_desugar_javac9_for_lint is not supported");
    }
    if (options.alwaysRewriteLongCompare) {
      throw new AssertionError("--rewrite_calls_to_long_compare has no effect");
    }
    if (!options.tolerateMissingDependencies) {
      throw new AssertionError("--best_effort_tolerate_missing_deps must be enabled");
    }
    if (options.desugarCoreLibs) {
      if (options.desugaredLibConfig == null) {
        throw new AssertionError(
            "If --desugar_supported_core_libs is set --desugared_lib_config "
                + " must also be set.");
      }
    }
    if (!options.desugarInterfaceMethodBodiesIfNeeded) {
      throw new AssertionError("--desugar_interface_method_bodies_if_needed must be enabled");
    }
    if (!options.desugarTryWithResourcesIfNeeded) {
      throw new AssertionError("--desugar_try_with_resources_if_needed must be enabled");
    }
    if (options.desugarTryWithResourcesOmitRuntimeClasses) {
      throw new AssertionError(
          "--desugar_try_with_resources_omit_runtime_classes is not supported");
    }
    if (options.generateBaseClassesForDefaultMethods) {
      throw new AssertionError("--generate_base_classes_for_default_methods is not supported");
    }
    if (options.copyBridgesFromClasspath) {
      throw new AssertionError("--copy_bridges_from_classpath is not supported");
    }
    if (options.coreLibrary) {
      throw new AssertionError("--core_library is not supported");
    }
    if (!options.rewriteCoreLibraryPrefixes.isEmpty()) {
      throw new AssertionError("--rewrite_core_library_prefix is not supported");
    }
    if (!options.emulateCoreLibraryInterfaces.isEmpty()) {
      throw new AssertionError("--emulate_core_library_interface is not supported");
    }
    if (!options.retargetCoreLibraryMembers.isEmpty()) {
      throw new AssertionError("--retarget_core_library_member is not supported");
    }
    if (!options.dontTouchCoreLibraryMembers.isEmpty()) {
      throw new AssertionError("--dont_rewrite_core_library_invocation is not supported");
    }
    if (!options.preserveCoreLibraryOverrides.isEmpty()) {
      throw new AssertionError("--preserve_core_library_override is not supported");
    }
    if (options.legacyJacocoFix) {
      throw new AssertionError("--legacy_jacoco_fix is not supported");
    }
    if (!options.desugarNestBasedPrivateAccess) {
      throw new AssertionError("--desugar_nest_based_private_access must be enabled");
    }
    if (!options.desugarIndifyStringConcat) {
      throw new AssertionError("--desugar_indy_string_concat must be enabled");
    }
    if (options.persistentWorker) {
      throw new AssertionError("--persistent_worker is not supported");
    }

    checkArgument(!options.inputJars.isEmpty(), "--input is required");
    checkArgument(
        options.inputJars.size() == options.outputJars.size(),
        "D8 Desugar requires the same number of inputs and outputs to pair them."
            + " #input=%s,#output=%s",
        options.inputJars.size(),
        options.outputJars.size());
  }

  public static void main(String[] args) throws Exception {
    DesugarOptions options = parseCommandLineOptions(args);
    validateOptions(options);

    new Desugar(options).desugar();
  }
}
