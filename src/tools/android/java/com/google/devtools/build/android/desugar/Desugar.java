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
import static com.google.devtools.build.android.desugar.strconcat.IndyStringConcatDesugaring.INVOKE_JDK11_STRING_CONCAT;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.ByteStreams;
import com.google.common.io.Closer;
import com.google.common.io.Resources;
import com.google.devtools.build.android.desugar.config.DesugarOptions;
import com.google.devtools.build.android.desugar.corelibadapter.InvocationSiteTransformationRecord;
import com.google.devtools.build.android.desugar.corelibadapter.InvocationSiteTransformationRecord.InvocationSiteTransformationRecordBuilder;
import com.google.devtools.build.android.desugar.corelibadapter.ShadowedApiAdaptersGenerator;
import com.google.devtools.build.android.desugar.corelibadapter.ShadowedApiInvocationSite;
import com.google.devtools.build.android.desugar.corelibadapter.ShadowedApiInvocationSite.ImmutableLabelRemover;
import com.google.devtools.build.android.desugar.covariantreturn.NioBufferRefConverter;
import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter;
import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter.UnprefixingClassWriter;
import com.google.devtools.build.android.desugar.io.FileBasedTypeReferenceClosure;
import com.google.devtools.build.android.desugar.io.FileContentProvider;
import com.google.devtools.build.android.desugar.io.HeaderClassLoader;
import com.google.devtools.build.android.desugar.io.IndexedInputs;
import com.google.devtools.build.android.desugar.io.InputFileProvider;
import com.google.devtools.build.android.desugar.io.JarDigest;
import com.google.devtools.build.android.desugar.io.OutputFileProvider;
import com.google.devtools.build.android.desugar.io.ResourceBasedClassFiles;
import com.google.devtools.build.android.desugar.io.ThrowingClassLoader;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberUseCounter;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.DesugarClassAttribute;
import com.google.devtools.build.android.desugar.nest.NestAnalyzer;
import com.google.devtools.build.android.desugar.nest.NestDesugaring;
import com.google.devtools.build.android.desugar.nest.NestDigest;
import com.google.devtools.build.android.desugar.strconcat.IndyStringConcatDesugaring;
import com.google.devtools.build.android.desugar.typeannotation.LocalTypeAnnotationUse;
import com.google.devtools.build.android.desugar.typehierarchy.TypeHierarchy;
import com.google.devtools.build.android.desugar.typehierarchy.TypeHierarchyScavenger;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
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
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.objectweb.asm.Attribute;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.tree.ClassNode;

/**
 * Command-line tool to desugar Java 8 constructs that dx doesn't know what to do with, in
 * particular lambdas and method references.
 */
public class Desugar {

  // It is important that this method is called first. See its javadoc.
  private static final Path DUMP_DIRECTORY = createAndRegisterLambdaDumpDirectory();
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final ResourceBasedClassFiles resourceBasedClassFiles =
      new ResourceBasedClassFiles();
  private static final String RUNTIME_LIB_PACKAGE =
      "com/google/devtools/build/android/desugar/runtime/";
  private static final Attribute[] customAttributes = {new DesugarClassAttribute()};

  private final DesugarOptions options;
  private final CoreLibraryRewriter rewriter;
  private final LambdaClassMaker lambdas;
  private final GeneratedClassStore store = new GeneratedClassStore();
  private final ClassMemberUseCounter classMemberUseCounter =
      new ClassMemberUseCounter(ConcurrentHashMultiset.create());
  private final Set<String> visitedExceptionTypes = new LinkedHashSet<>();
  /** The counter to record the times of try-with-resources desugaring is invoked. */
  private final AtomicInteger numOfTryWithResourcesInvoked = new AtomicInteger();
  /** The counter to record the times of UnsignedLongs desugaring is invoked. */
  private final AtomicInteger numOfUnsignedLongsInvoked = new AtomicInteger();
  /**
   * The counter to record the times of desugar runtime library's hashcode implementation is
   * invoked.
   */
  private final AtomicInteger numOfPrimitiveHashCodeInvoked = new AtomicInteger();

  private final boolean outputJava7;
  private final boolean allowDefaultMethods;
  private final boolean allowTryWithResources;
  private final boolean allowCallsToObjectsNonNull;
  private final boolean allowCallsToLongCompare;
  private final boolean allowCallsToLongUnsigned;
  private final boolean allowCallsToPrimitiveWrappers;
  /** An instance of Desugar is expected to be used ONLY ONCE */
  private boolean used;

  private TypeHierarchy typeHierarchy;

  private Desugar(DesugarOptions options, Path dumpDirectory) {
    this.options = options;
    this.rewriter = new CoreLibraryRewriter(options.coreLibrary ? ClassName.IN_PROCESS_LABEL : "");
    this.lambdas = new LambdaClassMaker(dumpDirectory);
    this.outputJava7 = options.minSdkVersion < 24;
    this.allowDefaultMethods =
        options.desugarInterfaceMethodBodiesIfNeeded || options.minSdkVersion >= 24;
    this.allowTryWithResources =
        !options.desugarTryWithResourcesIfNeeded || options.minSdkVersion >= 19;
    this.allowCallsToObjectsNonNull = options.minSdkVersion >= 19;
    this.allowCallsToLongCompare = options.minSdkVersion >= 19 && !options.alwaysRewriteLongCompare;
    this.allowCallsToLongUnsigned = options.minSdkVersion >= 26;
    this.allowCallsToPrimitiveWrappers = options.minSdkVersion >= 24;
    this.used = false;
  }

  private void desugar() throws Exception {
    checkState(!this.used, "This Desugar instance has been used. Please create another one.");
    this.used = true;

    List<Path> platformJars =
        options.classpath.stream()
            .filter(path -> path.toString().endsWith(".jar"))
            .filter(path -> JarDigest.fromPath(path).isPlatformJar())
            .collect(Collectors.toList());
    if (!platformJars.isEmpty()) {
      logger.atInfo().log("Platform Jars in class path added to boot class path: %s", platformJars);
      options.bootclasspath =
          ImmutableList.<Path>builder().addAll(options.bootclasspath).addAll(platformJars).build();
    }

    typeHierarchy =
        TypeHierarchyScavenger.analyze(
            ImmutableList.<Path>builder()
                .addAll(options.inputJars)
                .addAll(options.classpath)
                .addAll(options.bootclasspath)
                .build(),
            /* requireTypeResolutionComplete= */ false);

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
    Path inputPath = inputOutputPair.getInput(); // the jar
    Path outputPath = inputOutputPair.getOutput();
    checkArgument(
        Files.isDirectory(inputPath) || !Files.isDirectory(outputPath),
        "Input jar file requires an output jar file");

    try (OutputFileProvider outputFileProvider = OutputFileProvider.create(outputPath);
        InputFileProvider inputFiles = InputFileProvider.open(inputPath)) {
      DependencyCollector depsCollector = createDepsCollector();
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
      ClassVsInterface interfaceCache = new ClassVsInterface(classpathReader);
      final CoreLibrarySupport coreLibrarySupport =
          options.desugarCoreLibs
              ? new CoreLibrarySupport(
                  rewriter,
                  loader,
                  options.rewriteCoreLibraryPrefixes,
                  options.emulateCoreLibraryInterfaces,
                  options.retargetCoreLibraryMembers,
                  options.dontTouchCoreLibraryMembers)
              : null;

      InvocationSiteTransformationRecordBuilder callSiteTransCollector =
          InvocationSiteTransformationRecord.builder();

      desugarClassesInInput(
          inputFiles,
          outputFileProvider,
          loader,
          classpathReader,
          depsCollector,
          bootclasspathReader,
          coreLibrarySupport,
          interfaceCache,
          interfaceLambdaMethodCollector,
          callSiteTransCollector);

      desugarAndWriteDumpedLambdaClassesToOutput(
          outputFileProvider,
          loader,
          classpathReader,
          depsCollector,
          bootclasspathReader,
          coreLibrarySupport,
          interfaceCache,
          interfaceLambdaMethodCollector.build(),
          bridgeMethodReader,
          callSiteTransCollector);

      desugarAndWriteGeneratedClasses(
          outputFileProvider,
          loader,
          classpathReader,
          depsCollector,
          bootclasspathReader,
          coreLibrarySupport,
          callSiteTransCollector);

      copyRuntimeClasses(outputFileProvider, coreLibrarySupport);

      ShadowedApiAdaptersGenerator adaptersGenerator =
          ShadowedApiAdaptersGenerator.create(callSiteTransCollector.build());
      adaptersGenerator.getApiAdapters().sink(outputFileProvider);
      ImmutableSet.Builder<ClassName> typeConverters = ImmutableSet.builder();
      if (coreLibrarySupport != null) {
        typeConverters.addAll(coreLibrarySupport.usedTypeConverters());
      }
      typeConverters.addAll(adaptersGenerator.getTypeConverters());
      copyTypeConverterClasses(outputFileProvider, ImmutableSet.copyOf(typeConverters.build()));

      byte[] depsInfo = depsCollector.toByteArray();
      if (depsInfo != null) {
        outputFileProvider.write(OutputFileProvider.DESUGAR_DEPS_FILENAME, depsInfo);
      }
    }

    ImmutableMap<Path, LambdaInfo> lambdasLeftBehind = lambdas.drain();
    checkState(lambdasLeftBehind.isEmpty(), "Didn't process %s", lambdasLeftBehind);
    ImmutableMap<String, ClassNode> generatedLeftBehind = store.drain();
    checkState(generatedLeftBehind.isEmpty(), "Didn't process %s", generatedLeftBehind.keySet());
  }

  private static void copyTypeConverterClasses(
      OutputFileProvider outputFileProvider, ImmutableSet<ClassName> initialTypes) {
    FileBasedTypeReferenceClosure typeReferenceClosure =
        new FileBasedTypeReferenceClosure(
            ClassName::isInDesugarRuntimeLibrary, resourceBasedClassFiles);
    typeReferenceClosure.findReachableReferencedTypes(initialTypes).stream()
        .map(resourceBasedClassFiles::getContent)
        .forEach(fileContent -> fileContent.sink(outputFileProvider));
  }

  /**
   * Returns a dependency collector for use with a single input Jar. If {@link
   * DesugarOptions#emitDependencyMetadata} is set, this method instantiates the collector
   * reflectively to allow compiling and using the desugar tool without this mechanism.
   */
  private DependencyCollector createDepsCollector() {
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
        throw new IllegalStateException("Can't emit desugaring metadata as requested");
      }
    } else if (options.tolerateMissingDependencies) {
      return DependencyCollector.NoWriteCollectors.NOOP;
    } else {
      return DependencyCollector.NoWriteCollectors.FAIL_ON_MISSING;
    }
  }

  private void copyRuntimeClasses(
      OutputFileProvider outputFileProvider, @Nullable CoreLibrarySupport coreLibrarySupport) {
    // 1. Copy any runtime classes needed due to core library desugaring.
    if (coreLibrarySupport != null) {
      coreLibrarySupport.usedRuntimeHelpers().stream()
          .filter(className -> className.startsWith(RUNTIME_LIB_PACKAGE))
          .distinct()
          .forEach(
              className -> {
                // We want core libraries to remain self-contained, so fail if we get here.
                checkState(!options.coreLibrary, "Core library shouldn't depend on %s", className);
                try (InputStream stream =
                    Desugar.class.getClassLoader().getResourceAsStream(className + ".class")) {
                  outputFileProvider.write(
                      className + ".class",
                      ByteStreams.toByteArray(
                          checkNotNull(stream, "Resource Not Found for %s.", className)));
                } catch (IOException e) {
                  throw new IOError(e);
                }
              });
    }

    // 2. See if we rewrote Long.unsigned*
    if (numOfUnsignedLongsInvoked.get() > 0) {
      try (InputStream stream =
          Desugar.class
              .getClassLoader()
              .getResourceAsStream(
                  "com/google/devtools/build/android/desugar/runtime/UnsignedLongs.class")) {
        outputFileProvider.write(
            "com/google/devtools/build/android/desugar/runtime/UnsignedLongs.class",
            ByteStreams.toByteArray(stream));
      } catch (IOException e) {
        throw new IOError(e);
      }
    }

    // 3. See if we rewrote <primitive>.hashcode
    if (numOfPrimitiveHashCodeInvoked.get() > 0) {
      resourceBasedClassFiles
          .getContent(
              ClassName.create(
                  "com/google/devtools/build/android/desugar/runtime/PrimitiveHashcode"))
          .sink(outputFileProvider);
    }

    // 4. See if we need to copy StringConcats methods for Indify string desugaring.
    if (classMemberUseCounter.getMemberUseCount(INVOKE_JDK11_STRING_CONCAT) > 0) {
      String resourceName = "com/google/devtools/build/android/desugar/runtime/StringConcats.class";
      try (InputStream stream = Resources.getResource(resourceName).openStream()) {
        outputFileProvider.write(resourceName, ByteStreams.toByteArray(stream));
      } catch (IOException e) {
        throw new IOError(e);
      }
    }

    // 5. See if we need to copy try-with-resources runtime library
    if (allowTryWithResources || options.desugarTryWithResourcesOmitRuntimeClasses) {
      // try-with-resources statements are okay in the output jar.
      return;
    }
    if (numOfTryWithResourcesInvoked.get() <= 0) {
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
      DependencyCollector depsCollector,
      ClassReaderFactory bootclasspathReader,
      @Nullable CoreLibrarySupport coreLibrarySupport,
      ClassVsInterface interfaceCache,
      ImmutableSet.Builder<String> interfaceLambdaMethodCollector,
      InvocationSiteTransformationRecordBuilder callSiteRecord)
      throws IOException {

    ImmutableList<FileContentProvider<? extends InputStream>> inputFileContents =
        inputFiles.toInputFileStreams();
    NestDigest nestDigest = NestAnalyzer.analyzeNests(inputFileContents);
    // Apply core library type name remapping to the digest instance produced by the nest analyzer,
    // since the analysis-oriented nest analyzer visits core library classes without name remapping
    // as those transformation-oriented visitors.
    nestDigest = nestDigest.acceptTypeMapper(rewriter.getPrefixer());
    for (FileContentProvider<? extends InputStream> inputFileProvider :
        Iterables.concat(inputFileContents, nestDigest.getCompanionFileProviders())) {
      String inputFilename = inputFileProvider.getBinaryPathName();
      if ("module-info.class".equals(inputFilename)
          || (inputFilename.endsWith("/module-info.class")
              && Pattern.matches("META-INF/versions/[0-9]+/module-info.class", inputFilename))) {
        continue; // Drop module-info.class since it has no meaning on Android
      }
      if (OutputFileProvider.DESUGAR_DEPS_FILENAME.equals(inputFilename)) {
        // TODO(kmb): rule out that this happens or merge input file with what's in depsCollector
        continue; // skip as we're writing a new file like this at the end or don't want it
      }

      try (InputStream content = inputFileProvider.get()) {
        // We can write classes uncompressed since they need to be converted to .dex format
        // for Android anyways. Resources are written as they were in the input jar to avoid
        // any danger of accidentally uncompressed resources ending up in an .apk.  We also simply
        // copy classes from Desugar's runtime library, which we build so they need no desugaring.
        // The runtime library typically uses constructs we'd otherwise desugar, so it's easier
        // to just skip it should it appear as a regular input (for idempotency).
        if (inputFilename.endsWith(".class")
            && ClassName.fromClassFileName(inputFilename).isDesugarEligible()) {
          ClassReader reader = rewriter.reader(content);
          UnprefixingClassWriter writer = rewriter.writer(ClassWriter.COMPUTE_MAXS);
          ClassVisitor visitor =
              createClassVisitorsForClassesInInputs(
                  loader,
                  classpathReader,
                  depsCollector,
                  bootclasspathReader,
                  coreLibrarySupport,
                  interfaceCache,
                  interfaceLambdaMethodCollector,
                  writer,
                  reader,
                  nestDigest,
                  callSiteRecord);
          if (writer == visitor) {
            // Just copy the input if there are no rewritings
            outputFileProvider.write(inputFilename, reader.b);
          } else {
            reader.accept(visitor, customAttributes, 0);
            String filename = writer.getClassName() + ".class";
            checkState(
                (options.coreLibrary && coreLibrarySupport != null)
                    || filename.equals(inputFilename));
            outputFileProvider.write(filename, writer.toByteArray());
          }
        } else {
          // Most other files (and directories) we want to just copy, but...
          String outputFilename = inputFilename;
          if (options.coreLibrary && coreLibrarySupport != null && inputFilename.endsWith("/")) {
            // rename core library directories together with files in them
            outputFilename = coreLibrarySupport.renameCoreLibrary(inputFilename);
          } else if (coreLibrarySupport != null
              && !inputFilename.endsWith("/")
              && inputFilename.startsWith("META-INF/services/")) {
            // rename j.u.ServiceLoader files for renamed core libraries so they're found
            String serviceName = inputFilename.substring("META-INF/services/".length());
            if (!serviceName.contains("/")
                && coreLibrarySupport.isRenamedCoreLibrary(serviceName.replace('.', '/'))) {
              outputFilename =
                  "META-INF/services/"
                      + coreLibrarySupport
                          .renameCoreLibrary(serviceName.replace('.', '/'))
                          .replace('/', '.');
            }
          }
          outputFileProvider.copyFrom(inputFilename, inputFiles, outputFilename);
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
      DependencyCollector depsCollector,
      ClassReaderFactory bootclasspathReader,
      @Nullable CoreLibrarySupport coreLibrarySupport,
      ClassVsInterface interfaceCache,
      ImmutableSet<String> interfaceLambdaMethods,
      @Nullable ClassReaderFactory bridgeMethodReader,
      InvocationSiteTransformationRecordBuilder callSiteTransCollector)
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
        InvokeDynamicLambdaMethodCollector collector = new InvokeDynamicLambdaMethodCollector();
        reader.accept(
            collector, customAttributes, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
        ImmutableSet<MethodInfo> lambdaMethods = collector.getLambdaMethodsUsedInInvokeDynamics();
        checkState(
            lambdaMethods.isEmpty(),
            "Didn't expect to find lambda methods but found %s",
            lambdaMethods);
        UnprefixingClassWriter writer =
            rewriter.writer(ClassWriter.COMPUTE_MAXS /*for invoking bridges*/);
        ClassVisitor visitor =
            createClassVisitorsForDumpedLambdaClasses(
                loader,
                classpathReader,
                depsCollector,
                bootclasspathReader,
                coreLibrarySupport,
                interfaceCache,
                interfaceLambdaMethods,
                bridgeMethodReader,
                lambdaClass.getValue(),
                writer,
                reader,
                callSiteTransCollector);
        reader.accept(visitor, customAttributes, 0);
        checkState(
            (options.coreLibrary && coreLibrarySupport != null)
                || rewriter
                    .unprefix(lambdaClass.getValue().desiredInternalName())
                    .equals(writer.getClassName()));
        outputFileProvider.write(writer.getClassName() + ".class", writer.toByteArray());
      }
    }
  }

  private void desugarAndWriteGeneratedClasses(
      OutputFileProvider outputFileProvider,
      ClassLoader loader,
      @Nullable ClassReaderFactory classpathReader,
      DependencyCollector depsCollector,
      ClassReaderFactory bootclasspathReader,
      @Nullable CoreLibrarySupport coreLibrarySupport,
      InvocationSiteTransformationRecordBuilder callSiteTransCollector)
      throws IOException {
    // Write out any classes we generated along the way
    if (coreLibrarySupport != null) {
      coreLibrarySupport.makeDispatchHelpers(store);
    }
    ImmutableMap<String, ClassNode> generatedClasses = store.drain();
    checkState(
        generatedClasses.isEmpty() || (allowDefaultMethods && outputJava7),
        "Didn't expect generated classes but got %s",
        generatedClasses.keySet());
    for (Map.Entry<String, ClassNode> generated : generatedClasses.entrySet()) {
      UnprefixingClassWriter writer = rewriter.writer(ClassWriter.COMPUTE_MAXS);
      // checkState above implies that we want Java 7 .class files, so send through that visitor.
      // Don't need a ClassReaderFactory b/c static interface methods should've been moved.
      ClassVisitor visitor = writer;
      if (coreLibrarySupport != null) {
        visitor = new ImmutableLabelRemover(visitor);
        visitor = new EmulatedInterfaceRewriter(visitor, coreLibrarySupport);
        visitor = new CorePackageRenamer(visitor, coreLibrarySupport);
        visitor = new CoreLibraryInvocationRewriter(visitor, coreLibrarySupport);
        if (options.autoDesugarShadowedApiUse) {
          visitor = new ShadowedApiInvocationSite(visitor, callSiteTransCollector, typeHierarchy);
        }
      }

      if (!allowTryWithResources) {
        CloseResourceMethodScanner closeResourceMethodScanner = new CloseResourceMethodScanner();
        generated.getValue().accept(closeResourceMethodScanner);
        visitor =
            new TryWithResourcesRewriter(
                visitor,
                loader,
                visitedExceptionTypes,
                numOfTryWithResourcesInvoked,
                closeResourceMethodScanner.hasCloseResourceMethod());
      }
      if (!allowCallsToObjectsNonNull) {
        // Not sure whether there will be implicit null check emitted by javac, so we rerun
        // the inliner again
        visitor = new ObjectsRequireNonNullMethodRewriter(visitor, rewriter);
      }
      if (!allowCallsToLongUnsigned) {
        visitor = new LongUnsignedMethodRewriter(visitor, rewriter, numOfUnsignedLongsInvoked);
      }
      if (!allowCallsToLongCompare) {
        visitor = new LongCompareMethodRewriter(visitor, rewriter);
      }
      if (!allowCallsToPrimitiveWrappers) {
        visitor = new PrimitiveWrapperRewriter(visitor, numOfPrimitiveHashCodeInvoked);
      }

      visitor = new Java7Compatibility(visitor, (ClassReaderFactory) null, bootclasspathReader);
      if (options.generateBaseClassesForDefaultMethods) {
        // Use DefaultMethodClassFixer to make generated base classes extend other base classes if
        // possible and add any stubs from extended interfaces
        visitor =
            new DefaultMethodClassFixer(
                visitor,
                /*useGeneratedBaseClasses=*/ true,
                classpathReader,
                depsCollector,
                coreLibrarySupport,
                bootclasspathReader,
                loader);
      }
      generated.getValue().accept(visitor);
      checkState(
          (options.coreLibrary && coreLibrarySupport != null)
              || rewriter.unprefix(generated.getKey()).equals(writer.getClassName()));
      outputFileProvider.write(writer.getClassName() + ".class", writer.toByteArray());
    }
  }

  /**
   * Create the class visitors for the lambda classes that are generated on the fly. If no new class
   * visitors are not generated, then the passed-in {@code writer} will be returned.
   */
  private ClassVisitor createClassVisitorsForDumpedLambdaClasses(
      ClassLoader loader,
      @Nullable ClassReaderFactory classpathReader,
      DependencyCollector depsCollector,
      ClassReaderFactory bootclasspathReader,
      @Nullable CoreLibrarySupport coreLibrarySupport,
      ClassVsInterface interfaceCache,
      ImmutableSet<String> interfaceLambdaMethods,
      @Nullable ClassReaderFactory bridgeMethodReader,
      LambdaInfo lambdaClass,
      UnprefixingClassWriter writer,
      ClassReader input,
      InvocationSiteTransformationRecordBuilder callSiteRecord) {
    ClassVisitor visitor = checkNotNull(writer);

    if (coreLibrarySupport != null) {
      visitor = new ImmutableLabelRemover(visitor);
      visitor = new EmulatedInterfaceRewriter(visitor, coreLibrarySupport);
      visitor = new CorePackageRenamer(visitor, coreLibrarySupport);
      visitor = new CoreLibraryInvocationRewriter(visitor, coreLibrarySupport);
      if (options.autoDesugarShadowedApiUse) {
        visitor = new ShadowedApiInvocationSite(visitor, callSiteRecord, typeHierarchy);
      }
    }

    if (!allowTryWithResources) {
      CloseResourceMethodScanner closeResourceMethodScanner = new CloseResourceMethodScanner();
      input.accept(closeResourceMethodScanner, customAttributes, ClassReader.SKIP_DEBUG);
      visitor =
          new TryWithResourcesRewriter(
              visitor,
              loader,
              visitedExceptionTypes,
              numOfTryWithResourcesInvoked,
              closeResourceMethodScanner.hasCloseResourceMethod());
    }
    if (!allowCallsToObjectsNonNull) {
      // Not sure whether there will be implicit null check emitted by javac, so we rerun
      // the inliner again
      visitor = new ObjectsRequireNonNullMethodRewriter(visitor, rewriter);
    }
    if (!allowCallsToLongUnsigned) {
      visitor = new LongUnsignedMethodRewriter(visitor, rewriter, numOfUnsignedLongsInvoked);
    }
    if (!allowCallsToLongCompare) {
      visitor = new LongCompareMethodRewriter(visitor, rewriter);
    }
    if (!allowCallsToPrimitiveWrappers) {
      visitor = new PrimitiveWrapperRewriter(visitor, numOfPrimitiveHashCodeInvoked);
    }
    if (outputJava7) {
      // null ClassReaderFactory b/c we don't expect to need it for lambda classes
      visitor = new Java7Compatibility(visitor, (ClassReaderFactory) null, bootclasspathReader);
      if (options.desugarInterfaceMethodBodiesIfNeeded) {
        visitor =
            new DefaultMethodClassFixer(
                visitor,
                options.generateBaseClassesForDefaultMethods,
                classpathReader,
                depsCollector,
                coreLibrarySupport,
                bootclasspathReader,
                loader);
        visitor =
            new InterfaceDesugaring(
                visitor,
                options.generateBaseClassesForDefaultMethods,
                interfaceCache,
                depsCollector,
                coreLibrarySupport,
                bootclasspathReader,
                loader,
                store,
                options.legacyJacocoFix);
      }
    }

    visitor =
        new LambdaClassFixer(
            visitor,
            lambdaClass,
            bridgeMethodReader,
            loader,
            interfaceLambdaMethods,
            allowDefaultMethods,
            outputJava7);
    // Send lambda classes through desugaring to make sure there's no invokedynamic
    // instructions in generated lambda classes (checkState below will fail)
    visitor =
        new LambdaDesugaring(
            visitor, loader, lambdas, null, ImmutableSet.of(), allowDefaultMethods);
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
      DependencyCollector depsCollector,
      ClassReaderFactory bootclasspathReader,
      @Nullable CoreLibrarySupport coreLibrarySupport,
      ClassVsInterface interfaceCache,
      ImmutableSet.Builder<String> interfaceLambdaMethodCollector,
      UnprefixingClassWriter writer,
      ClassReader input,
      NestDigest nestDigest,
      InvocationSiteTransformationRecordBuilder callSiteRecord) {
    ClassVisitor visitor = checkNotNull(writer);


    if (coreLibrarySupport != null) {
      visitor = new ImmutableLabelRemover(visitor);
      visitor = new EmulatedInterfaceRewriter(visitor, coreLibrarySupport);
      visitor = new CorePackageRenamer(visitor, coreLibrarySupport);
      visitor = new CoreLibraryInvocationRewriter(visitor, coreLibrarySupport);
      if (options.autoDesugarShadowedApiUse) {
        visitor = new ShadowedApiInvocationSite(visitor, callSiteRecord, typeHierarchy);
      }
    }

    if (!allowTryWithResources) {
      CloseResourceMethodScanner closeResourceMethodScanner = new CloseResourceMethodScanner();
      input.accept(closeResourceMethodScanner, customAttributes, ClassReader.SKIP_DEBUG);
      visitor =
          new TryWithResourcesRewriter(
              visitor,
              loader,
              visitedExceptionTypes,
              numOfTryWithResourcesInvoked,
              closeResourceMethodScanner.hasCloseResourceMethod());
    }
    if (!allowCallsToObjectsNonNull) {
      visitor = new ObjectsRequireNonNullMethodRewriter(visitor, rewriter);
    }
    if (!allowCallsToLongUnsigned) {
      visitor = new LongUnsignedMethodRewriter(visitor, rewriter, numOfUnsignedLongsInvoked);
    }
    if (!allowCallsToLongCompare) {
      visitor = new LongCompareMethodRewriter(visitor, rewriter);
    }
    if (!allowCallsToPrimitiveWrappers) {
      visitor = new PrimitiveWrapperRewriter(visitor, numOfPrimitiveHashCodeInvoked);
    }
    if (!options.onlyDesugarJavac9ForLint) {
      if (outputJava7) {
        visitor = new Java7Compatibility(visitor, classpathReader, bootclasspathReader);
        if (options.desugarInterfaceMethodBodiesIfNeeded) {
          visitor =
              new DefaultMethodClassFixer(
                  visitor,
                  options.generateBaseClassesForDefaultMethods,
                  classpathReader,
                  depsCollector,
                  coreLibrarySupport,
                  bootclasspathReader,
                  loader);
          visitor =
              new InterfaceDesugaring(
                  visitor,
                  options.generateBaseClassesForDefaultMethods,
                  interfaceCache,
                  depsCollector,
                  coreLibrarySupport,
                  bootclasspathReader,
                  loader,
                  store,
                  options.legacyJacocoFix);
        }
      }

      // LambdaDesugaring is relatively expensive, so check first whether we need it.  Additionally,
      // we need to collect lambda methods referenced by invokedynamic instructions up-front anyway.
      // TODO(kmb): Scan constant pool instead of visiting the class to find bootstrap methods etc.
      InvokeDynamicLambdaMethodCollector collector = new InvokeDynamicLambdaMethodCollector();
      input.accept(collector, customAttributes, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
      ImmutableSet<MethodInfo> methodsUsedInInvokeDynamics =
          collector.getLambdaMethodsUsedInInvokeDynamics();
      if (!methodsUsedInInvokeDynamics.isEmpty() || collector.needOuterClassRewrite()) {
        visitor =
            new LambdaDesugaring(
                visitor,
                loader,
                lambdas,
                interfaceLambdaMethodCollector,
                methodsUsedInInvokeDynamics,
                allowDefaultMethods);
      }
    }

    if (options.desugarNestBasedPrivateAccess) {
      visitor = new NestDesugaring(visitor, nestDigest);
    }

    if (options.desugarIndifyStringConcat) {
      visitor = new IndyStringConcatDesugaring(classMemberUseCounter, visitor);
    }

    visitor = NioBufferRefConverter.create(visitor, rewriter.getPrefixer());

    visitor = new LocalTypeAnnotationUse(visitor);

    return visitor;
  }

  public static void main(String[] args) throws Exception {
    verifyLambdaDumpDirectoryRegistered(DUMP_DIRECTORY);

    DesugarOptions options = DesugarOptions.parseCommandLineOptions(args);
    if (options.persistentWorker) {
      runPersistentWorker(DUMP_DIRECTORY);
    } else {
      processRequest(options, DUMP_DIRECTORY);
    }
  }

  private static void runPersistentWorker(Path dumpDirectory) throws Exception {
    while (true) {
      WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);

      if (request == null) {
        break;
      }

      String[] argList = new String[request.getArgumentsCount()];
      argList = request.getArgumentsList().toArray(argList);

      DesugarOptions options = DesugarOptions.parseCommandLineOptions(argList);

      try {
        processRequest(options, dumpDirectory);
        logger.atInfo().log(
            "Processing Request success: %s", WorkResponse.newBuilder().setExitCode(0).build());
      } catch (Exception e) {
        logger.atWarning().withCause(e).log(
            "Processing Request exception: %s",
            WorkResponse.newBuilder().setExitCode(1).setOutput(e.getMessage()).build());
      }
    }
  }

  private static int processRequest(DesugarOptions options, Path dumpDirectory) throws Exception {
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
    checkArgument(
        !options.desugarCoreLibs
            || !options.rewriteCoreLibraryPrefixes.isEmpty()
            || !options.emulateCoreLibraryInterfaces.isEmpty(),
        "--desugar_supported_core_libs requires specifying renamed and/or emulated core libraries");

    if (options.verbose) {
      logger.atInfo().log("Lambda classes will be written under %s%n", dumpDirectory);
    }
    new Desugar(options, dumpDirectory).desugar();

    return 0;
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
      logger.atWarning().withCause(e).log(
          "Failed to verify lambda dump directory due to unavailable class/field access. \n"
              + "Continue desugaring...");
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
  static Path createAndRegisterLambdaDumpDirectory() {
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

    Path dumpDirectory;
    try {
      dumpDirectory = Files.createTempDirectory("lambdas");
    } catch (IOException e) {
      throw new IOError(e);
    }
    System.setProperty(LAMBDA_METAFACTORY_DUMPER_PROPERTY, dumpDirectory.toString());
    deleteTreeOnExit(dumpDirectory);
    return dumpDirectory;
  }

  private static ImmutableList<InputOutputPair> toInputOutputPairs(DesugarOptions options) {
    final ImmutableList.Builder<InputOutputPair> ioPairListbuilder = ImmutableList.builder();
    for (Iterator<Path> inputIt = options.inputJars.iterator(),
            outputIt = options.outputJars.iterator();
        inputIt.hasNext(); ) {
      ioPairListbuilder.add(InputOutputPair.create(inputIt.next(), outputIt.next()));
    }
    return ioPairListbuilder.build();
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
      builder.add(closer.register(InputFileProvider.open(path)));
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
