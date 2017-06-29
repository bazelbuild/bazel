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
package com.google.devtools.build.android.desugar;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.android.desugar.LambdaClassMaker.LAMBDA_METAFACTORY_DUMPER_PROPERTY;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSet.Builder;
import com.google.common.io.ByteStreams;
import com.google.common.io.Closer;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.desugar.CoreLibraryRewriter.UnprefixingClassWriter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
import com.google.devtools.common.options.proto.OptionFilters.OptionEffectTag;
import com.google.errorprone.annotations.MustBeClosed;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.tree.ClassNode;

/**
 * Command-line tool to desugar Java 8 constructs that dx doesn't know what to do with, in
 * particular lambdas and method references.
 */
class Desugar {

  /** Commandline options for {@link Desugar}. */
  public static class Options extends OptionsBase {
    @Option(
      name = "input",
      allowMultiple = true,
      defaultValue = "",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      abbrev = 'i',
      help =
          "Input Jar or directory with classes to desugar (required, the n-th input is paired with"
              + "the n-th output)."
    )
    public List<Path> inputJars;

    @Option(
      name = "classpath_entry",
      allowMultiple = true,
      defaultValue = "",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      help =
          "Ordered classpath (Jar or directory) to resolve symbols in the --input Jar, like "
              + "javac's -cp flag."
    )
    public List<Path> classpath;

    @Option(
      name = "bootclasspath_entry",
      allowMultiple = true,
      defaultValue = "",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      help =
          "Bootclasspath that was used to compile the --input Jar with, like javac's "
              + "-bootclasspath flag (required)."
    )
    public List<Path> bootclasspath;

    @Option(
      name = "allow_empty_bootclasspath",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED
    )
    public boolean allowEmptyBootclasspath;

    @Option(
      name = "only_desugar_javac9_for_lint",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "A temporary flag specifically for android lint, subject to removal anytime (DO NOT USE)",
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED
    )
    public boolean onlyDesugarJavac9ForLint;

    @Option(
      name = "rewrite_calls_to_long_compare",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Rewrite calls to Long.compare(long, long) to the JVM instruction lcmp "
              + "regardless of --min_sdk_version.",
      category = "misc"
    )
    public boolean alwaysRewriteLongCompare;

    @Option(
      name = "output",
      allowMultiple = true,
      defaultValue = "",
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = PathConverter.class,
      abbrev = 'o',
      help =
          "Output Jar or directory to write desugared classes into (required, the n-th output is "
              + "paired with the n-th input, output must be a Jar if input is a Jar)."
    )
    public List<Path> outputJars;

    @Option(
      name = "verbose",
      defaultValue = "false",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      abbrev = 'v',
      help = "Enables verbose debugging output."
    )
    public boolean verbose;

    @Option(
      name = "min_sdk_version",
      defaultValue = "1",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Minimum targeted sdk version.  If >= 24, enables default methods in interfaces."
    )
    public int minSdkVersion;

    @Option(
      name = "desugar_interface_method_bodies_if_needed",
      defaultValue = "true",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Rewrites default and static methods in interfaces if --min_sdk_version < 24. This "
              + "only works correctly if subclasses of rewritten interfaces as well as uses of "
              + "static interface methods are run through this tool as well."
    )
    public boolean desugarInterfaceMethodBodiesIfNeeded;

    @Option(
      name = "desugar_try_with_resources_if_needed",
      defaultValue = "true",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Rewrites try-with-resources statements if --min_sdk_version < 19."
    )
    public boolean desugarTryWithResourcesIfNeeded;

    @Option(
      name = "desugar_try_with_resources_omit_runtime_classes",
      defaultValue = "false",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Omits the runtime classes necessary to support try-with-resources from the output. "
              + "This property has effect only if --desugar_try_with_resources_if_needed is used."
    )
    public boolean desugarTryWithResourcesOmitRuntimeClasses;

    @Option(
      name = "copy_bridges_from_classpath",
      defaultValue = "false",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Copy bridges from classpath to desugared classes."
    )
    public boolean copyBridgesFromClasspath;

    @Option(
      name = "core_library",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
      implicitRequirements = "--allow_empty_bootclasspath",
      help = "Enables rewriting to desugar java.* classes."
    )
    public boolean coreLibrary;
  }

  private final Options options;
  private final CoreLibraryRewriter rewriter;
  private final LambdaClassMaker lambdas;
  private final GeneratedClassStore store;
  private final Set<String> visitedExceptionTypes = new HashSet<>();
  /** The counter to record the times of try-with-resources desugaring is invoked. */
  private final AtomicInteger numOfTryWithResourcesInvoked = new AtomicInteger();

  private final boolean outputJava7;
  private final boolean allowDefaultMethods;
  private final boolean allowCallsToObjectsNonNull;
  private final boolean allowCallsToLongCompare;
  /** An instance of Desugar is expected to be used ONLY ONCE */
  private boolean used;

  private Desugar(Options options, Path dumpDirectory) {
    this.options = options;
    this.rewriter = new CoreLibraryRewriter(options.coreLibrary ? "__desugar__/" : "");
    this.lambdas = new LambdaClassMaker(dumpDirectory);
    this.store = new GeneratedClassStore();
    this.outputJava7 = options.minSdkVersion < 24;
    this.allowDefaultMethods =
        options.desugarInterfaceMethodBodiesIfNeeded || options.minSdkVersion >= 24;
    this.allowCallsToObjectsNonNull = options.minSdkVersion >= 19;
    this.allowCallsToLongCompare = options.minSdkVersion >= 19 && !options.alwaysRewriteLongCompare;
    this.used = false;
  }

  private void desugar() throws Exception {
    checkState(!this.used, "This Desugar instance has been used. Please create another one.");
    this.used = true;

    try (Closer closer = Closer.create()) {
      IndexedInputs indexedBootclasspath =
          new IndexedInputs(toRegisteredInputFileProvider(closer, options.bootclasspath));
      // Use a classloader that as much as possible uses the provided bootclasspath instead of
      // the tool's system classloader.  Unfortunately we can't do that for java. classes.
      ClassLoader bootclassloader =
          options.bootclasspath.isEmpty()
              ? new ThrowingClassLoader()
              : new HeaderClassLoader(indexedBootclasspath, rewriter, new ThrowingClassLoader());
      IndexedInputs indexedClasspath =
          new IndexedInputs(toRegisteredInputFileProvider(closer, options.classpath));

      // Process each input separately
      for (InputOutputPair inputOutputPair : toInputOutputPairs(options)) {
        desugarOneInput(
            inputOutputPair,
            indexedClasspath,
            bootclassloader,
            new ClassReaderFactory(indexedBootclasspath, rewriter));
      }
    }
  }

  private void desugarOneInput(
      InputOutputPair inputOutputPair,
      IndexedInputs indexedClasspath,
      ClassLoader bootclassloader,
      ClassReaderFactory bootclasspathReader)
      throws Exception {
    Path inputPath = inputOutputPair.getInput();
    Path outputPath = inputOutputPair.getOutput();
    checkArgument(
        Files.isDirectory(inputPath) || !Files.isDirectory(outputPath),
        "Input jar file requires an output jar file");

    try (OutputFileProvider outputFileProvider = toOutputFileProvider(outputPath);
        InputFileProvider inputFiles = toInputFileProvider(inputPath)) {
      IndexedInputs indexedInputFiles = new IndexedInputs(ImmutableList.of(inputFiles));
      // Prepend classpath with input file itself so LambdaDesugaring can load classes with
      // lambdas.
      IndexedInputs indexedClasspathAndInputFiles = indexedClasspath.withParent(indexedInputFiles);
      // Note that input file and classpath need to be in the same classloader because
      // we typically get the header Jar for inputJar on the classpath and having the header
      // Jar in a parent loader means the header version is preferred over the real thing.
      ClassLoader loader =
          new HeaderClassLoader(indexedClasspathAndInputFiles, rewriter, bootclassloader);

      ClassReaderFactory classpathReader = null;
      ClassReaderFactory bridgeMethodReader = null;
      if (outputJava7) {
        classpathReader = new ClassReaderFactory(indexedClasspathAndInputFiles, rewriter);
        if (options.copyBridgesFromClasspath) {
          bridgeMethodReader = classpathReader;
        } else {
          bridgeMethodReader = new ClassReaderFactory(indexedInputFiles, rewriter);
        }
      }

      ImmutableSet.Builder<String> interfaceLambdaMethodCollector = ImmutableSet.builder();

      desugarClassesInInput(
          inputFiles,
          outputFileProvider,
          loader,
          classpathReader,
          bootclasspathReader,
          interfaceLambdaMethodCollector);

      desugarAndWriteDumpedLambdaClassesToOutput(
          outputFileProvider,
          loader,
          classpathReader,
          bootclasspathReader,
          interfaceLambdaMethodCollector.build(),
          bridgeMethodReader);

      desugarAndWriteGeneratedClasses(outputFileProvider);
      copyThrowableExtensionClass(outputFileProvider);
    }

    ImmutableMap<Path, LambdaInfo> lambdasLeftBehind = lambdas.drain();
    checkState(lambdasLeftBehind.isEmpty(), "Didn't process %s", lambdasLeftBehind);
    ImmutableMap<String, ClassNode> generatedLeftBehind = store.drain();
    checkState(generatedLeftBehind.isEmpty(), "Didn't process %s", generatedLeftBehind.keySet());
  }

  private void copyThrowableExtensionClass(OutputFileProvider outputFileProvider) {
    if (!outputJava7
        || !options.desugarTryWithResourcesIfNeeded
        || options.desugarTryWithResourcesOmitRuntimeClasses) {
      // try-with-resources statements are okay in the output jar.
      return;
    }
    if (this.numOfTryWithResourcesInvoked.get() <= 0) {
      // the try-with-resources desugaring pass does nothing, so no need to copy these class files.
      return;
    }
    for (String className :
        TryWithResourcesRewriter.THROWABLE_EXT_CLASS_INTERNAL_NAMES_WITH_CLASS_EXT) {
      try (InputStream stream = Desugar.class.getClassLoader().getResourceAsStream(className)) {
        outputFileProvider.write(className, ByteStreams.toByteArray(stream));
      } catch (IOException e) {
        throw new IOError(e);
      }
    }
  }

  /** Desugar the classes that are in the inputs specified in the command line arguments. */
  private void desugarClassesInInput(
      InputFileProvider inputFiles,
      OutputFileProvider outputFileProvider,
      ClassLoader loader,
      @Nullable ClassReaderFactory classpathReader,
      ClassReaderFactory bootclasspathReader,
      Builder<String> interfaceLambdaMethodCollector)
      throws IOException {
    for (String filename : inputFiles) {
      try (InputStream content = inputFiles.getInputStream(filename)) {
        // We can write classes uncompressed since they need to be converted to .dex format
        // for Android anyways. Resources are written as they were in the input jar to avoid
        // any danger of accidentally uncompressed resources ending up in an .apk.
        if (filename.endsWith(".class")) {
          ClassReader reader = rewriter.reader(content);
          UnprefixingClassWriter writer = rewriter.writer(ClassWriter.COMPUTE_MAXS);
          ClassVisitor visitor =
              createClassVisitorsForClassesInInputs(
                  loader,
                  classpathReader,
                  bootclasspathReader,
                  interfaceLambdaMethodCollector,
                  writer);
          reader.accept(visitor, 0);

          outputFileProvider.write(filename, writer.toByteArray());
        } else {
          outputFileProvider.copyFrom(filename, inputFiles);
        }
      }
    }
  }

  /**
   * Desugar the classes that are generated on the fly when we are desugaring the classes in the
   * specified inputs.
   */
  private void desugarAndWriteDumpedLambdaClassesToOutput(
      OutputFileProvider outputFileProvider,
      ClassLoader loader,
      @Nullable ClassReaderFactory classpathReader,
      ClassReaderFactory bootclasspathReader,
      ImmutableSet<String> interfaceLambdaMethods,
      @Nullable ClassReaderFactory bridgeMethodReader)
      throws IOException {
    checkState(
        !allowDefaultMethods || interfaceLambdaMethods.isEmpty(),
        "Desugaring with default methods enabled moved interface lambdas");

    // Write out the lambda classes we generated along the way
    ImmutableMap<Path, LambdaInfo> lambdaClasses = lambdas.drain();
    checkState(
        !options.onlyDesugarJavac9ForLint || lambdaClasses.isEmpty(),
        "There should be no lambda classes generated: %s",
        lambdaClasses.keySet());

    for (Map.Entry<Path, LambdaInfo> lambdaClass : lambdaClasses.entrySet()) {
      try (InputStream bytecode = Files.newInputStream(lambdaClass.getKey())) {
        ClassReader reader = rewriter.reader(bytecode);
        UnprefixingClassWriter writer =
            rewriter.writer(ClassWriter.COMPUTE_MAXS /*for invoking bridges*/);
        ClassVisitor visitor =
            createClassVisitorsForDumpedLambdaClasses(
                loader,
                classpathReader,
                bootclasspathReader,
                interfaceLambdaMethods,
                bridgeMethodReader,
                lambdaClass.getValue(),
                writer);
        reader.accept(visitor, 0);
        String filename =
            rewriter.unprefix(lambdaClass.getValue().desiredInternalName()) + ".class";
        outputFileProvider.write(filename, writer.toByteArray());
      }
    }
  }

  private void desugarAndWriteGeneratedClasses(OutputFileProvider outputFileProvider)
      throws IOException {
    // Write out any classes we generated along the way
    ImmutableMap<String, ClassNode> generatedClasses = store.drain();
    checkState(
        generatedClasses.isEmpty() || (allowDefaultMethods && outputJava7),
        "Didn't expect generated classes but got %s",
        generatedClasses.keySet());
    for (Map.Entry<String, ClassNode> generated : generatedClasses.entrySet()) {
      UnprefixingClassWriter writer = rewriter.writer(ClassWriter.COMPUTE_MAXS);
      // checkState above implies that we want Java 7 .class files, so send through that visitor.
      // Don't need a ClassReaderFactory b/c static interface methods should've been moved.
      ClassVisitor visitor = new Java7Compatibility(writer, (ClassReaderFactory) null);
      generated.getValue().accept(visitor);
      String filename = rewriter.unprefix(generated.getKey()) + ".class";
      outputFileProvider.write(filename, writer.toByteArray());
    }
  }

  /**
   * Create the class visitors for the lambda classes that are generated on the fly. If no new class
   * visitors are not generated, then the passed-in {@code writer} will be returned.
   */
  private ClassVisitor createClassVisitorsForDumpedLambdaClasses(
      ClassLoader loader,
      @Nullable ClassReaderFactory classpathReader,
      ClassReaderFactory bootclasspathReader,
      ImmutableSet<String> interfaceLambdaMethods,
      @Nullable ClassReaderFactory bridgeMethodReader,
      LambdaInfo lambdaClass,
      UnprefixingClassWriter writer) {
    ClassVisitor visitor = checkNotNull(writer);

    if (outputJava7) {
      // null ClassReaderFactory b/c we don't expect to need it for lambda classes
      visitor = new Java7Compatibility(visitor, (ClassReaderFactory) null);
      if (options.desugarTryWithResourcesIfNeeded) {
        visitor =
            new TryWithResourcesRewriter(
                visitor, loader, visitedExceptionTypes, numOfTryWithResourcesInvoked);
      }
      if (options.desugarInterfaceMethodBodiesIfNeeded) {
        visitor =
            new DefaultMethodClassFixer(visitor, classpathReader, bootclasspathReader, loader);
        visitor = new InterfaceDesugaring(visitor, bootclasspathReader, store);
      }
    }
    visitor =
        new LambdaClassFixer(
            visitor,
            lambdaClass,
            bridgeMethodReader,
            interfaceLambdaMethods,
            allowDefaultMethods,
            outputJava7);
    // Send lambda classes through desugaring to make sure there's no invokedynamic
    // instructions in generated lambda classes (checkState below will fail)
    visitor = new LambdaDesugaring(visitor, loader, lambdas, null, allowDefaultMethods);
    if (!allowCallsToObjectsNonNull) {
      // Not sure whether there will be implicit null check emitted by javac, so we rerun
      // the inliner again
      visitor = new ObjectsRequireNonNullMethodRewriter(visitor);
    }
    if (!allowCallsToLongCompare) {
      visitor = new LongCompareMethodRewriter(visitor);
    }
    return visitor;
  }

  /**
   * Create the class visitors for the classes which are in the inputs. If new visitors are created,
   * then all these visitors and the passed-in writer will be chained together. If no new visitor is
   * created, then the passed-in {@code writer} will be returned.
   */
  private ClassVisitor createClassVisitorsForClassesInInputs(
      ClassLoader loader,
      @Nullable ClassReaderFactory classpathReader,
      ClassReaderFactory bootclasspathReader,
      Builder<String> interfaceLambdaMethodCollector,
      UnprefixingClassWriter writer) {
    ClassVisitor visitor = checkNotNull(writer);

    if (!options.onlyDesugarJavac9ForLint) {
      if (outputJava7) {
        visitor = new Java7Compatibility(visitor, classpathReader);
        if (options.desugarTryWithResourcesIfNeeded) {
          visitor =
              new TryWithResourcesRewriter(
                  visitor, loader, visitedExceptionTypes, numOfTryWithResourcesInvoked);
        }
        if (options.desugarInterfaceMethodBodiesIfNeeded) {
          visitor =
              new DefaultMethodClassFixer(visitor, classpathReader, bootclasspathReader, loader);
          visitor = new InterfaceDesugaring(visitor, bootclasspathReader, store);
        }
      }
      visitor =
          new LambdaDesugaring(
              visitor, loader, lambdas, interfaceLambdaMethodCollector, allowDefaultMethods);
    }

    if (!allowCallsToObjectsNonNull) {
      visitor = new ObjectsRequireNonNullMethodRewriter(visitor);
    }
    if (!allowCallsToLongCompare) {
      visitor = new LongCompareMethodRewriter(visitor);
    }
    return visitor;
  }

  public static void main(String[] args) throws Exception {
    // It is important that this method is called first. See its javadoc.
    Path dumpDirectory = createAndRegisterLambdaDumpDirectory();
    verifyLambdaDumpDirectoryRegistered(dumpDirectory);

    Options options = parseCommandLineOptions(args);
    if (options.verbose) {
      System.out.printf("Lambda classes will be written under %s%n", dumpDirectory);
    }
    new Desugar(options, dumpDirectory).desugar();
  }

  static void verifyLambdaDumpDirectoryRegistered(Path dumpDirectory) throws IOException {
    try {
      Class<?> klass = Class.forName("java.lang.invoke.InnerClassLambdaMetafactory");
      Field dumperField = klass.getDeclaredField("dumper");
      dumperField.setAccessible(true);
      Object dumperValue = dumperField.get(null);
      checkNotNull(dumperValue, "Failed to register lambda dump directory '%s'", dumpDirectory);

      Field dumperPathField = dumperValue.getClass().getDeclaredField("dumpDir");
      dumperPathField.setAccessible(true);
      Object dumperPath = dumperPathField.get(dumperValue);
      checkState(
          dumperPath instanceof Path && Files.isSameFile(dumpDirectory, (Path) dumperPath),
          "Inconsistent lambda dump directories. real='%s', expected='%s'",
          dumperPath,
          dumpDirectory);
    } catch (ReflectiveOperationException e) {
      // We do not want to crash Desugar, if we cannot load or access these classes or fields.
      // We aim to provide better diagnostics. If we cannot, just let it go.
      e.printStackTrace();
    }
  }

  /**
   * LambdaClassMaker generates lambda classes for us, but it does so by essentially simulating the
   * call to LambdaMetafactory that the JVM would make when encountering an invokedynamic.
   * LambdaMetafactory is in the JDK and its implementation has a property to write out ("dump")
   * generated classes, which we take advantage of here. This property can be set externally, and in
   * that case the specified directory is used as a temporary dir. Otherwise, it will be set here,
   * before doing anything else since the property is read in the static initializer.
   */
  static Path createAndRegisterLambdaDumpDirectory() throws IOException {
    String propertyValue = System.getProperty(LAMBDA_METAFACTORY_DUMPER_PROPERTY);
    if (propertyValue != null) {
      Path path = Paths.get(propertyValue);
      checkState(Files.isDirectory(path), "The path '%s' is not a directory.", path);
      // It is not necessary to check whether 'path' is an empty directory. It is possible that
      // LambdaMetafactory is loaded before this class, and there are already lambda classes dumped
      // into the 'path' folder.
      // TODO(kmb): Maybe we can empty the folder here.
      return path;
    }

    Path dumpDirectory = Files.createTempDirectory("lambdas");
    System.setProperty(LAMBDA_METAFACTORY_DUMPER_PROPERTY, dumpDirectory.toString());
    deleteTreeOnExit(dumpDirectory);
    return dumpDirectory;
  }

  private static Options parseCommandLineOptions(String[] args) throws IOException {
    if (args.length == 1 && args[0].startsWith("@")) {
      args = Files.readAllLines(Paths.get(args[0].substring(1)), ISO_8859_1).toArray(new String[0]);
    }

    OptionsParser optionsParser = OptionsParser.newOptionsParser(Options.class);
    optionsParser.setAllowResidue(false);
    optionsParser.parseAndExitUponError(args);

    Options options = optionsParser.getOptions(Options.class);

    checkArgument(!options.inputJars.isEmpty(), "--input is required");
    checkArgument(
        options.inputJars.size() == options.outputJars.size(),
        "Desugar requires the same number of inputs and outputs to pair them. #input=%s,#output=%s",
        options.inputJars.size(),
        options.outputJars.size());
    checkArgument(
        !options.bootclasspath.isEmpty() || options.allowEmptyBootclasspath,
        "At least one --bootclasspath_entry is required");
    for (Path path : options.bootclasspath) {
      checkArgument(!Files.isDirectory(path), "Bootclasspath entry must be a jar file: %s", path);
    }
    return options;
  }

  private static ImmutableList<InputOutputPair> toInputOutputPairs(Options options) {
    final ImmutableList.Builder<InputOutputPair> ioPairListbuilder = ImmutableList.builder();
    for (Iterator<Path> inputIt = options.inputJars.iterator(),
            outputIt = options.outputJars.iterator();
        inputIt.hasNext(); ) {
      ioPairListbuilder.add(InputOutputPair.create(inputIt.next(), outputIt.next()));
    }
    return ioPairListbuilder.build();
  }

  @VisibleForTesting
  static class ThrowingClassLoader extends ClassLoader {
    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
      if (name.startsWith("java.")) {
        // Use system class loader for java. classes, since ClassLoader.defineClass gets
        // grumpy when those don't come from the standard place.
        return super.loadClass(name, resolve);
      }
      throw new ClassNotFoundException();
    }
  }

  private static void deleteTreeOnExit(final Path directory) {
    Thread shutdownHook =
        new Thread() {
          @Override
          public void run() {
            try {
              deleteTree(directory);
            } catch (IOException e) {
              throw new RuntimeException("Failed to delete " + directory, e);
            }
          }
        };
    Runtime.getRuntime().addShutdownHook(shutdownHook);
  }

  /** Recursively delete a directory. */
  private static void deleteTree(final Path directory) throws IOException {
    if (directory.toFile().exists()) {
      Files.walkFileTree(
          directory,
          new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
                throws IOException {
              Files.delete(file);
              return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc)
                throws IOException {
              Files.delete(dir);
              return FileVisitResult.CONTINUE;
            }
          });
    }
  }

  /** Transform a Path to an {@link OutputFileProvider} */
  @MustBeClosed
  private static OutputFileProvider toOutputFileProvider(Path path) throws IOException {
    if (Files.isDirectory(path)) {
      return new DirectoryOutputFileProvider(path);
    } else {
      return new ZipOutputFileProvider(path);
    }
  }

  /** Transform a Path to an InputFileProvider that needs to be closed by the caller. */
  @MustBeClosed
  private static InputFileProvider toInputFileProvider(Path path) throws IOException {
    if (Files.isDirectory(path)) {
      return new DirectoryInputFileProvider(path);
    } else {
      return new ZipInputFileProvider(path);
    }
  }

  /**
   * Transform a list of Path to a list of InputFileProvider and register them with the given
   * closer.
   */
  @SuppressWarnings("MustBeClosedChecker")
  @VisibleForTesting
  static ImmutableList<InputFileProvider> toRegisteredInputFileProvider(
      Closer closer, List<Path> paths) throws IOException {
    ImmutableList.Builder<InputFileProvider> builder = new ImmutableList.Builder<>();
    for (Path path : paths) {
      builder.add(closer.register(toInputFileProvider(path)));
    }
    return builder.build();
  }

  /** Pair input and output. */
  @AutoValue
  abstract static class InputOutputPair {

    static InputOutputPair create(Path input, Path output) {
      return new AutoValue_Desugar_InputOutputPair(input, output);
    }

    abstract Path getInput();

    abstract Path getOutput();
  }
}
