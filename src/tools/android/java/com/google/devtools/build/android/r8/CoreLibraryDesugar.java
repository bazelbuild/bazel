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
import com.android.tools.r8.Diagnostic;
import com.android.tools.r8.DiagnosticsHandler;
import com.android.tools.r8.L8;
import com.android.tools.r8.L8Command;
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
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** CoreLibraryDesugar compatible wrapper based on D8 desugaring engine */
public class CoreLibraryDesugar {

  /** Commandline options for {@link CoreLibraryDesugar}. */
  public static class DesugarOptions extends OptionsBase {

    @Option(
        name = "input",
        allowMultiple = false,
        defaultValue = "null",
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = ExistingPathConverter.class,
        abbrev = 'i',
        help = "Input jar with classes to desugar.")
    public Path inputJar;

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
            "Bootclasspath that was used to compile the --input jars with, like javac's "
                + "-bootclasspath flag (required).")
    public List<Path> bootclasspath;

    @Option(
        name = "output",
        allowMultiple = false,
        defaultValue = "null",
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = PathConverter.class,
        abbrev = 'o',
        help = "Output jar to write desugared classes into.")
    public Path outputJar;

    @Option(
        name = "min_sdk_version",
        defaultValue = "1",
        category = "misc",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Minimum targeted sdk version.  If >= 24, enables default methods in interfaces.")
    public int minSdkVersion;

    @Option(
        name = "desugar_supported_core_libs",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Enable core library desugaring, which requires configuration with related flags.")
    public boolean desugarCoreLibs;

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

  private CoreLibraryDesugar(DesugarOptions options) {
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
    return parser.getOptions(DesugarOptions.class);
  }

  private class DesugarDiagnosticsHandler implements DiagnosticsHandler {

    OutputConsumer outputConsumer;

    private DesugarDiagnosticsHandler(OutputConsumer outputConsumer) {
      this.outputConsumer = outputConsumer;
    }

    @Override
    public void warning(Diagnostic warning) {
      // Workaround for b/181634110.
      if (warning instanceof InterfaceDesugarMissingTypeDiagnostic) {
        InterfaceDesugarMissingTypeDiagnostic missingTypeDiagnostic =
            (InterfaceDesugarMissingTypeDiagnostic) warning;
        if (missingTypeDiagnostic.getMissingType().getTypeName().equals("jdk.internal.misc.Unsafe")
            && missingTypeDiagnostic
                .getContextType()
                .getTypeName()
                .equals("java.util.concurrent.ThreadLocalRandomHelper")) {
          return;
        }
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
    DependencyCollector dependencyCollector = DependencyCollector.NoWriteCollectors.FAIL_ON_MISSING;
    OutputConsumer consumer = new OutputConsumer(output, dependencyCollector, input);
    L8Command.Builder builder =
        L8Command.builder(new DesugarDiagnosticsHandler(consumer))
            .addClasspathResourceProvider(classpath)
            .addProgramFiles(input)
            .setMinApiLevel(options.minSdkVersion)
            .setProgramConsumer(consumer);
    bootclasspathProviders.forEach(builder::addLibraryResourceProvider);
    if (desugaredLibConfig != null) {
      builder.addDesugaredLibraryConfiguration(StringResource.fromFile(desugaredLibConfig));
    }
    L8.run(builder.build());
  }

  private void desugar() throws CompilationFailedException, IOException {
    ImmutableList.Builder<ClassFileResourceProvider> bootclasspathProvidersBuilder =
        ImmutableList.builder();
    for (Path path : options.bootclasspath) {
      bootclasspathProvidersBuilder.add(new ArchiveClassFileProvider(path));
    }
    ImmutableList.Builder<ClassFileResourceProvider> classpathProvidersBuilder =
        ImmutableList.builder();
    for (Path path : options.classpath) {
      classpathProvidersBuilder.add(new ArchiveClassFileProvider(path));
    }

    ImmutableList<ClassFileResourceProvider> bootclasspathProviders =
        bootclasspathProvidersBuilder.build();
    OrderedClassFileResourceProvider classpathProvider =
        new OrderedClassFileResourceProvider(
            bootclasspathProviders, classpathProvidersBuilder.build());

    // Desugar the input core library code.
    desugar(
        bootclasspathProviders,
        classpathProvider,
        options.inputJar,
        options.outputJar,
        options.desugaredLibConfig);
  }

  private static void validateOptions(DesugarOptions options) {
    if (!options.desugarCoreLibs || options.desugaredLibConfig == null) {
      throw new AssertionError(
          "Both options --desugar_supported_core_libs and --desugared_lib_config"
              + " must be passed.");
    }
  }

  public static void main(String[] args) throws Exception {
    DesugarOptions options = parseCommandLineOptions(args);
    validateOptions(options);

    new CoreLibraryDesugar(options).desugar();
  }
}
