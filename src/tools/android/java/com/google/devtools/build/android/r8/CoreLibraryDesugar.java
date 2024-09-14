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

import static com.android.tools.r8.CompilationMode.RELEASE;
import static com.google.common.base.Preconditions.checkArgument;
import static com.google.devtools.build.android.r8.desugar.OutputConsumer.Flags.EXCLUDE_PATH_ENTRIES;

import com.android.tools.r8.ArchiveClassFileProvider;
import com.android.tools.r8.ClassFileResourceProvider;
import com.android.tools.r8.CompilationFailedException;
import com.android.tools.r8.Diagnostic;
import com.android.tools.r8.DiagnosticsHandler;
import com.android.tools.r8.L8;
import com.android.tools.r8.L8Command;
import com.android.tools.r8.errors.InterfaceDesugarMissingTypeDiagnostic;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidOptionsUtils;
import com.google.devtools.build.android.r8.CompatOptionsConverters.CompatExistingPathConverter;
import com.google.devtools.build.android.r8.CompatOptionsConverters.CompatPathConverter;
import com.google.devtools.build.android.r8.desugar.OrderedClassFileResourceProvider;
import com.google.devtools.build.android.r8.desugar.OutputConsumer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** CoreLibraryDesugar compatible wrapper based on D8 desugaring engine */
public class CoreLibraryDesugar {

  /** Commandline options for {@link CoreLibraryDesugar}. */
  @Parameters(separators = "= ")
  public static class DesugarOptions {

    @Parameter(
        names = {"--input", "-i"},
        converter = CompatExistingPathConverter.class,
        description = "Input jar with classes to desugar.")
    public Path inputJar;

    @Parameter(
        names = "--classpath_entry",
        converter = CompatExistingPathConverter.class,
        description =
            "Ordered classpath (Jar or directory) to resolve symbols in the --input Jar, like "
                + "javac's -cp flag.")
    public List<Path> classpath = ImmutableList.of();

    @Parameter(
        names = "--bootclasspath_entry",
        converter = CompatExistingPathConverter.class,
        description =
            "Bootclasspath that was used to compile the --input jars with, like javac's "
                + "-bootclasspath flag (required).")
    public List<Path> bootclasspath = ImmutableList.of();

    @Parameter(
        names = {"--output", "-o"},
        converter = CompatPathConverter.class,
        description = "Output jar to write desugared classes into.")
    public Path outputJar;

    @Parameter(
        names = "--min_sdk_version",
        description =
            "Minimum targeted sdk version.  If >= 24, enables default methods in interfaces.")
    public int minSdkVersion = Integer.parseInt(Constants.MIN_API_LEVEL);

    @Parameter(
        names = "--desugar_supported_core_libs",
        arity = 1,
        description =
            "Enable core library desugaring, which requires configuration with related flags.")
    public boolean desugarCoreLibs;

    @Parameter(
        names = "--desugared_lib_config",
        converter = CompatExistingPathConverter.class,
        description =
            "Specify desugared library configuration. "
                + "The input file is a desugared library configuration (json)")
    public Path desugaredLibConfig;
  }

  private final DesugarOptions options;

  private CoreLibraryDesugar(DesugarOptions options) {
    this.options = options;
  }

  private static DesugarOptions parseCommandLineOptions(String[] args) {
    DesugarOptions options = new DesugarOptions();
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(options, preprocessedArgs);
    JCommander.newBuilder().addObject(options).build().parse(normalizedArgs);
    return options;
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
      throws CompilationFailedException, IOException {
    checkArgument(!Files.isDirectory(input), "Input must be a jar (%s is a directory)", input);
    DependencyCollector dependencyCollector = NoWriteCollectors.FAIL_ON_MISSING;
    OutputConsumer consumer =
        new OutputConsumer(output, dependencyCollector, input, EXCLUDE_PATH_ENTRIES);
    L8Command.Builder builder =
        L8Command.builder(new DesugarDiagnosticsHandler(consumer))
            .addClasspathResourceProvider(classpath)
            .addProgramFiles(input)
            .setMinApiLevel(options.minSdkVersion)
            .setMode(RELEASE)
            .setProgramConsumer(consumer);
    bootclasspathProviders.forEach(builder::addLibraryResourceProvider);
    if (desugaredLibConfig != null) {
      builder.addDesugaredLibraryConfiguration(Files.readString(desugaredLibConfig));
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
