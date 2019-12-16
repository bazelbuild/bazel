/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar;

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.annotations.UsedReflectively;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.runtime.RunfilesPaths;
import com.google.errorprone.annotations.FormatMethod;
import java.io.IOException;
import java.io.InputStream;
import java.lang.annotation.Annotation;
import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.rules.TemporaryFolder;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.model.Statement;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;

/** A JUnit4 Rule that desugars an input jar file and load the transformed jar to JVM. */
public class DesugarRule implements TestRule {

  /**
   * Identifies injectable class-literal fields with the specified class to load at runtime and
   * assign to the field. An injectable class-literal field may have any access modifier (private,
   * package-private, protected, public). Sample usage:
   *
   * <pre><code>
   * &#064;RunWith(JUnit4.class)
   * public class DesugarRuleTest {
   *
   *   &#064;Rule
   *   public final DesugarRule desugarRule =
   *       DesugarRule.builder(this, MethodHandles.lookup())
   *           .addRuntimeInputs("path/to/my_jar.jar")
   *           .build();
   *
   *   &#064;LoadClass("my.package.ClassToDesugar")
   *   private Class<?> classToDesugarClass;
   *
   *   // ... Test methods ...
   * }
   * </code></pre>
   */
  @UsedReflectively
  @Documented
  @Target(ElementType.FIELD)
  @Retention(RetentionPolicy.RUNTIME)
  public @interface LoadClass {

    /**
     * The fully-qualified class name of the class to load. The format agrees with {@link
     * Class#getName}.
     */
    String value();

    /** The round during which its associated jar is being used. */
    int round() default 1;
  }

  /**
   * Identifies injectable {@link ZipEntry} fields with a zip entry path. The desugar rule resolves
   * the requested zip entry at runtime and assign it to the annotated field. An injectable {@link
   * ZipEntry} field may have any access modifier (private, package-private, protected, public).
   * Sample usage:
   *
   * <pre><code>
   * &#064;RunWith(JUnit4.class)
   * public class DesugarRuleTest {
   *
   *   &#064;Rule
   *   public final DesugarRule desugarRule =
   *       DesugarRule.builder(this, MethodHandles.lookup())
   *           .addRuntimeInputs("path/to/my_jar.jar")
   *           .build();
   *
   *   &#064;LoadZipEntry("my/package/ClassToDesugar.class")
   *   private ZipEntry classToDesugarClassFile;
   *
   *   // ... Test methods ...
   * }
   * </code></pre>
   */
  @UsedReflectively
  @Documented
  @Target(ElementType.FIELD)
  @Retention(RetentionPolicy.RUNTIME)
  public @interface LoadZipEntry {

    /** The requested zip entry path name within a zip file. */
    String value();

    /** The round during which its associated jar is being used. */
    int round() default 1;
  }

  /**
   * Identifies injectable ASM node fields (e.g. {@link org.objectweb.asm.tree.ClassNode}, {@link
   * org.objectweb.asm.tree.MethodNode}, {@link org.objectweb.asm.tree.FieldNode}) with a qualified
   * class name. The desugar rule resolves the requested class at runtime, parse it into a {@link
   * ClassNode} and assign parsed class node to the annotated field. An injectable ASM node field
   * may have any access modifier (private, package-private, protected, public). Sample usage:
   *
   * <pre><code>
   * &#064;RunWith(JUnit4.class)
   * public class DesugarRuleTest {
   *
   *   &#064;Rule
   *   public final DesugarRule desugarRule =
   *       DesugarRule.builder(this, MethodHandles.lookup())
   *           .addRuntimeInputs("path/to/my_jar.jar")
   *           .build();
   *
   *   &#064;LoadAsmNode("my.package.ClassToDesugar")
   *   private ClassNode classToDesugarClassFile;
   *
   *   // ... Test methods ...
   * }
   * </code></pre>
   */
  @UsedReflectively
  @Documented
  @Target(ElementType.FIELD)
  @Retention(RetentionPolicy.RUNTIME)
  public @interface LoadAsmNode {

    /**
     * The fully-qualified class name of the class to load. The format agrees with {@link
     * Class#getName}.
     */
    String className();

    /** If non-empty, load the specified class member (field or method) from the enclosing class. */
    String memberName() default "";

    /** If non-empty, use the specified member descriptor to disambiguate overloaded methods. */
    String memberDescriptor() default "";

    /** The round during which its associated jar is being used. */
    int round() default 1;
  }

  private static final Path ANDROID_RUNTIME_JAR_PATH =
      RunfilesPaths.resolve(
          "third_party/java/android/android_sdk_linux/platforms/experimental/android_blaze.jar");

  private static final Path JACOCO_RUNTIME_PATH =
      RunfilesPaths.resolve("third_party/java/jacoco/jacoco_agent.jar");

  private static final String DEFAULT_OUTPUT_ROOT_PREFIX = "desugared_dump";

  private static final ClassLoader BASE_CLASS_LOADER =
      ClassLoader.getSystemClassLoader().getParent();

  /** The transformation record that describes the desugaring of a jar. */
  @AutoValue
  abstract static class JarTransformationRecord {

    /**
     * The full runtime path of a pre-transformationRecord jar.
     *
     * @see Desugar.DesugarOptions#inputJars for details.
     */
    abstract ImmutableList<Path> inputJars();

    /**
     * The full runtime path of a post-transformationRecord jar (deguared jar).
     *
     * @see Desugar.DesugarOptions#inputJars for details.
     */
    abstract ImmutableList<Path> outputJars();

    /** @see Desugar.DesugarOptions#classpath for details. */
    abstract ImmutableList<Path> classPathEntries();

    /** @see Desugar.DesugarOptions#bootclasspath for details. */
    abstract ImmutableList<Path> bootClassPathEntries();

    /** The remaining command options used for desugaring. */
    abstract ImmutableListMultimap<String, String> extraCustomCommandOptions();

    /** The factory method of this jar transformation record. */
    static JarTransformationRecord create(
        ImmutableList<Path> inputJars,
        ImmutableList<Path> outputJars,
        ImmutableList<Path> classPathEntries,
        ImmutableList<Path> bootClassPathEntries,
        ImmutableListMultimap<String, String> extraCustomCommandOptions) {
      return new AutoValue_DesugarRule_JarTransformationRecord(
          inputJars, outputJars, classPathEntries, bootClassPathEntries, extraCustomCommandOptions);
    }

    final ImmutableList<String> getDesugarFlags() {
      ImmutableList.Builder<String> args = ImmutableList.builder();
      inputJars().forEach(path -> args.add("--input=" + path));
      outputJars().forEach(path -> args.add("--output=" + path));
      classPathEntries().forEach(path -> args.add("--classpath_entry=" + path));
      bootClassPathEntries().forEach(path -> args.add("--bootclasspath_entry=" + path));
      extraCustomCommandOptions().forEach((k, v) -> args.add("--" + k + "=" + v));
      return args.build();
    }

    @Memoized
    ClassLoader getOutputClassLoader() throws MalformedURLException {
      List<URL> urls = new ArrayList<>();
      for (Path path : Iterables.concat(outputJars(), classPathEntries(), bootClassPathEntries())) {
        urls.add(path.toUri().toURL());
      }
      return URLClassLoader.newInstance(urls.toArray(new URL[0]), BASE_CLASS_LOADER);
    }
  }

  /** For hosting desugared jar temporarily. */
  private final TemporaryFolder temporaryFolder = new TemporaryFolder();

  private final Object testInstance;
  private final MethodHandles.Lookup testInstanceLookup;

  /** The maximum number of desugar operations, used for testing idempotency. */
  private final int maxNumOfTransformations;

  private final List<JarTransformationRecord> jarTransformationRecords;

  private final ImmutableList<Path> inputs;
  private final ImmutableList<Path> classPathEntries;
  private final ImmutableList<Path> bootClassPathEntries;
  private final ImmutableListMultimap<String, String> extraCustomCommandOptions;

  private final ImmutableList<Field> injectableClassLiterals;
  private final ImmutableList<Field> injectableAsmNodes;
  private final ImmutableList<Field> injectableZipEntries;

  /** The state of the already-created directories to avoid directory re-creation. */
  private final Map<String, Path> tempDirs = new HashMap<>();

  /**
   * The entry point to create a {@link DesugarRule}.
   *
   * @param testInstance The <code>this</code> reference of the JUnit test class.
   * @param testInstanceLookup The lookup object from the test class, i.e.<code>
   *     MethodHandles.lookup()</code>
   */
  public static DesugarRuleBuilder builder(Object testInstance, Lookup testInstanceLookup) {
    return new DesugarRuleBuilder(testInstance, testInstanceLookup);
  }

  private DesugarRule(
      Object testInstance,
      Lookup testInstanceLookup,
      int maxNumOfTransformations,
      List<JarTransformationRecord> jarTransformationRecords,
      ImmutableList<Path> bootClassPathEntries,
      ImmutableListMultimap<String, String> customCommandOptions,
      ImmutableList<Path> inputJars,
      ImmutableList<Path> classPathEntries,
      ImmutableList<Field> injectableClassLiterals,
      ImmutableList<Field> injectableAsmNodes,
      ImmutableList<Field> injectableZipEntries) {
    this.testInstance = testInstance;
    this.testInstanceLookup = testInstanceLookup;

    this.maxNumOfTransformations = maxNumOfTransformations;
    this.jarTransformationRecords = jarTransformationRecords;

    this.inputs = inputJars;
    this.classPathEntries = classPathEntries;
    this.bootClassPathEntries = bootClassPathEntries;
    this.extraCustomCommandOptions = customCommandOptions;

    this.injectableClassLiterals = injectableClassLiterals;
    this.injectableAsmNodes = injectableAsmNodes;
    this.injectableZipEntries = injectableZipEntries;
  }

  @Override
  public Statement apply(Statement base, Description description) {
    return temporaryFolder.apply(
        new Statement() {
          @Override
          public void evaluate() throws Throwable {
            ImmutableList<Path> transInputs = inputs;
            for (int round = 1; round <= maxNumOfTransformations; round++) {
              ImmutableList<Path> transOutputs =
                  getRuntimeOutputPaths(
                      transInputs,
                      temporaryFolder,
                      tempDirs,
                      /* outputRootPrefix= */ DEFAULT_OUTPUT_ROOT_PREFIX + "_" + round);
              JarTransformationRecord transformationRecord =
                  JarTransformationRecord.create(
                      transInputs,
                      transOutputs,
                      classPathEntries,
                      bootClassPathEntries,
                      extraCustomCommandOptions);
              Desugar.main(transformationRecord.getDesugarFlags().toArray(new String[0]));

              jarTransformationRecords.add(transformationRecord);
              transInputs = transOutputs;
            }

            for (Field field : injectableClassLiterals) {
              Class<?> classLiteral =
                  getClassLiteral(
                      field.getDeclaredAnnotation(LoadClass.class),
                      getInputClassLoader(),
                      jarTransformationRecords);
              MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
              fieldSetter.invoke(testInstance, classLiteral);
            }

            for (Field field : injectableAsmNodes) {
              Class<?> requestedFieldType = field.getType();
              Object asmNode =
                  getAsmNode(
                      field.getDeclaredAnnotation(LoadAsmNode.class),
                      requestedFieldType,
                      jarTransformationRecords,
                      inputs);
              MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
              fieldSetter.invoke(testInstance, asmNode);
            }

            for (Field field : injectableZipEntries) {
              ZipEntry zipEntry = getZipEntry(field.getDeclaredAnnotation(LoadZipEntry.class));
              MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
              fieldSetter.invoke(testInstance, zipEntry);
            }
            base.evaluate();
          }
        },
        description);
  }

  private static Class<?> getClassLiteral(
      LoadClass classLiteralRequest,
      ClassLoader initialInputClassLoader,
      List<JarTransformationRecord> jarTransformationRecords)
      throws Throwable {
    String qualifiedClassName = classLiteralRequest.value();
    int round = classLiteralRequest.round();
    ClassLoader outputJarClassLoader =
        round == 0
            ? initialInputClassLoader
            : jarTransformationRecords.get(round - 1).getOutputClassLoader();
    return outputJarClassLoader.loadClass(qualifiedClassName);
  }

  private static Object getAsmNode(
      LoadAsmNode asmNodeRequest,
      Class<?> requestedNodeType,
      List<JarTransformationRecord> jarTransformationRecords,
      ImmutableList<Path> initialInputs)
      throws IOException, ClassNotFoundException {
    String qualifiedClassName = asmNodeRequest.className();
    String classFileName = qualifiedClassName.replace('.', '/') + ".class";
    int round = asmNodeRequest.round();
    ImmutableList<Path> jars =
        round == 0 ? initialInputs : jarTransformationRecords.get(round - 1).outputJars();
    ClassNode classNode = findClassNode(classFileName, jars);
    if (requestedNodeType == ClassNode.class) {
      return classNode;
    }

    String memberName = asmNodeRequest.memberName();
    String memberDescriptor = asmNodeRequest.memberDescriptor();
    if (requestedNodeType == FieldNode.class) {
      return getFieldNode(classNode, memberName, memberDescriptor);
    }
    if (requestedNodeType == MethodNode.class) {
      return getMethodNode(classNode, memberName, memberDescriptor);
    }

    throw new UnsupportedOperationException(
        String.format("Injecting a node type (%s) is not supported", requestedNodeType));
  }

  private static FieldNode getFieldNode(
      ClassNode classNode, String fieldName, String fieldDescriptor) {
    for (FieldNode field : classNode.fields) {
      if (fieldName.equals(field.name)) {
        checkState(
            fieldDescriptor.isEmpty() || fieldDescriptor.equals(field.desc),
            "Found name-matched field but with a different descriptor. Expected requested field"
                + " descriptor, if specified, agrees with the actual field type. Field name <%s>"
                + " in class <%s>; Requested Type <%s>; Actual Type: <%s>.",
            fieldName,
            classNode.name,
            fieldDescriptor,
            field.desc);
        return field;
      }
    }
    throw new IllegalStateException(
        String.format("Field <%s> not found in class <%s>", fieldName, classNode.name));
  }

  private static MethodNode getMethodNode(
      ClassNode classNode, String methodName, String methodDescriptor) {
    boolean hasMethodDescriptor = !methodDescriptor.isEmpty();
    List<MethodNode> matchedMethods =
        classNode.methods.stream()
            .filter(methodNode -> methodName.equals(methodNode.name))
            .filter(methodNode -> !hasMethodDescriptor || methodDescriptor.equals(methodNode.desc))
            .collect(Collectors.toList());
    if (matchedMethods.isEmpty()) {
      throw new IllegalStateException(
          String.format(
              "Method <name:%s%s> is not found in class <%s>",
              methodName,
              hasMethodDescriptor ? ", descriptor:" + methodDescriptor : "",
              classNode.name));
    }
    if (matchedMethods.size() > 1) {
      List<String> matchedMethodDescriptors =
          matchedMethods.stream().map(method -> method.desc).collect(Collectors.toList());
      throw new IllegalStateException(
          String.format(
              "Multiple matches for requested method (name: %s in class %s). Please specify the"
                  + " method descriptor to disambiguate overloaded method request. All descriptors"
                  + " of name-matched methods: %s.",
              methodName, classNode.name, matchedMethodDescriptors));
    }
    return Iterables.getOnlyElement(matchedMethods);
  }

  private ZipEntry getZipEntry(LoadZipEntry zipEntryRequest) throws IOException {
    String zipEntryPathName = zipEntryRequest.value();
    int round = zipEntryRequest.round();
    ImmutableList<Path> jars =
        round == 0 ? inputs : jarTransformationRecords.get(round - 1).outputJars();
    for (Path jar : jars) {
      ZipFile zipFile = new ZipFile(jar.toFile());
      ZipEntry zipEntry = zipFile.getEntry(zipEntryPathName);
      if (zipEntry != null) {
        return zipEntry;
      }
    }
    throw new IllegalStateException(
        String.format("Expected zip entry of (%s) present.", zipEntryPathName));
  }

  private static ClassNode findClassNode(String zipEntryPathName, ImmutableList<Path> jars)
      throws IOException, ClassNotFoundException {
    for (Path jar : jars) {
      ZipFile zipFile = new ZipFile(jar.toFile());
      ZipEntry zipEntry = zipFile.getEntry(zipEntryPathName);
      if (zipEntry != null) {
        try (InputStream inputStream = zipFile.getInputStream(zipEntry)) {
          ClassReader cr = new ClassReader(inputStream);
          ClassNode classNode = new ClassNode(Opcodes.ASM7);
          cr.accept(classNode, 0);
          return classNode;
        }
      }
    }
    throw new ClassNotFoundException(zipEntryPathName);
  }

  private ClassLoader getInputClassLoader() throws MalformedURLException {
    List<URL> urls = new ArrayList<>();
    for (Path path : Iterables.concat(inputs, classPathEntries, bootClassPathEntries)) {
      urls.add(path.toUri().toURL());
    }
    return URLClassLoader.newInstance(urls.toArray(new URL[0]), BASE_CLASS_LOADER);
  }

  private static ImmutableList<Path> getRuntimeOutputPaths(
      ImmutableList<Path> inputs,
      TemporaryFolder temporaryFolder,
      Map<String, Path> tempDirs,
      String outputRootPrefix)
      throws IOException {
    ImmutableList.Builder<Path> outputRuntimePathsBuilder = ImmutableList.builder();
    for (Path path : inputs) {
      String targetDirKey = Paths.get(outputRootPrefix) + "/" + path.getParent();
      final Path outputDirPath;
      if (tempDirs.containsKey(targetDirKey)) {
        outputDirPath = tempDirs.get(targetDirKey);
      } else {
        outputDirPath = Paths.get(temporaryFolder.newFolder(targetDirKey).getPath());
        tempDirs.put(targetDirKey, outputDirPath);
      }
      outputRuntimePathsBuilder.add(outputDirPath.resolve(path.getFileName()));
    }
    return outputRuntimePathsBuilder.build();
  }

  private static ImmutableList<Field> findAllFieldsWithAnnotation(
      Class<?> testClass, Class<? extends Annotation> annotationClass) {
    ImmutableList.Builder<Field> fields = ImmutableList.builder();
    for (Class<?> currentClass = testClass;
        currentClass != null;
        currentClass = currentClass.getSuperclass()) {
      for (Field field : currentClass.getDeclaredFields()) {
        if (field.isAnnotationPresent(annotationClass)) {
          fields.add(field);
        }
      }
    }
    return fields.build();
  }

  /** The builder class for {@link DesugarRule}. */
  public static class DesugarRuleBuilder {

    private static final ImmutableSet<Class<?>> SUPPORTED_ASM_NODE_TYPES =
        ImmutableSet.of(ClassNode.class, FieldNode.class, MethodNode.class);

    private final Object testInstance;
    private final MethodHandles.Lookup testInstanceLookup;
    private final ImmutableList<Field> injectableClassLiterals;
    private final ImmutableList<Field> injectableAsmNodes;
    private final ImmutableList<Field> injectableJarFileEntries;
    private int maxNumOfTransformations = 1;
    private final List<Path> inputs = new ArrayList<>();
    private final List<Path> classPathEntries = new ArrayList<>();
    private final List<Path> bootClassPathEntries = new ArrayList<>();
    private final Multimap<String, String> customCommandOptions = LinkedHashMultimap.create();
    private final ErrorMessenger errorMessenger = new ErrorMessenger();

    DesugarRuleBuilder(Object testInstance, MethodHandles.Lookup testInstanceLookup) {
      this.testInstance = testInstance;
      this.testInstanceLookup = testInstanceLookup;
      Class<?> testClass = testInstance.getClass();
      if (testClass != testInstanceLookup.lookupClass()) {
        errorMessenger.addError(
            "Expected testInstanceLookup has private access to (%s), but get (%s). Have you"
                + " passed MethodHandles.lookup() to testInstanceLookup in test class?",
            testClass, testInstanceLookup.lookupClass());
      }
      if (!testClass.isAnnotationPresent(RunWith.class)) {
        errorMessenger.addError(
            "Expected a test instance whose class is annotated with @RunWith. %s", testClass);
      }

      injectableClassLiterals = findAllFieldsWithAnnotation(testClass, LoadClass.class);
      injectableAsmNodes = findAllFieldsWithAnnotation(testClass, LoadAsmNode.class);
      injectableJarFileEntries = findAllFieldsWithAnnotation(testClass, LoadZipEntry.class);
    }

    public DesugarRuleBuilder enableIterativeTransformation(int maxNumOfTransformations) {
      this.maxNumOfTransformations = maxNumOfTransformations;
      return this;
    }

    public DesugarRuleBuilder addRuntimeInputs(String... inputJars) {
      Arrays.stream(inputJars).map(RunfilesPaths::resolve).forEach(this::addInputs);
      return this;
    }

    public DesugarRuleBuilder addInputs(Path... inputJars) {
      for (Path path : inputJars) {
        if (!path.toString().endsWith(".jar")) {
          errorMessenger.addError("Expected a JAR file (*.jar): Actual (%s)", path);
        }
        if (!Files.exists(path)) {
          errorMessenger.addError("File does not exist: %s", path);
        }
        inputs.add(path);
      }
      return this;
    }

    public DesugarRuleBuilder addClasspathEntries(Path... inputJars) {
      Collections.addAll(classPathEntries, inputJars);
      return this;
    }

    public DesugarRuleBuilder addBootClassPathEntries(Path... inputJars) {
      Collections.addAll(bootClassPathEntries, inputJars);
      return this;
    }

    /** Format: --<key>=<value> */
    public DesugarRuleBuilder addCommandOptions(String key, String value) {
      customCommandOptions.put(key, value);
      return this;
    }

    private void checkJVMOptions() {
      RuntimeMXBean runtimeMxBean = ManagementFactory.getRuntimeMXBean();
      List<String> arguments = runtimeMxBean.getInputArguments();
      if (arguments.stream()
          .noneMatch(arg -> arg.startsWith("-Djdk.internal.lambda.dumpProxyClasses="))) {
        errorMessenger.addError(
            "Expected \"-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)\" in jvm_flags.\n");
      }
    }

    public DesugarRule build() {
      checkJVMOptions();
      checkInjectableClassLiterals();
      checkInjectableAsmNodes();
      checkInjectableZipEntries();

      if (bootClassPathEntries.isEmpty()
          && !customCommandOptions.containsKey("allow_empty_bootclasspath")) {
        addCommandOptions("bootclasspath_entry", ANDROID_RUNTIME_JAR_PATH.toString());
      }

      if (errorMessenger.containsAnyError()) {
        throw new IllegalStateException(
            String.format(
                "Invalid Desugar configurations:\n%s\n",
                String.join("\n", errorMessenger.getAllMessages())));
      }

      addClasspathEntries(JACOCO_RUNTIME_PATH);

      return new DesugarRule(
          testInstance,
          testInstanceLookup,
          maxNumOfTransformations,
          new ArrayList<>(maxNumOfTransformations),
          ImmutableList.copyOf(bootClassPathEntries),
          ImmutableListMultimap.copyOf(customCommandOptions),
          ImmutableList.copyOf(inputs),
          ImmutableList.copyOf(classPathEntries),
          injectableClassLiterals,
          injectableAsmNodes,
          injectableJarFileEntries);
    }

    private void checkInjectableClassLiterals() {
      for (Field field : injectableClassLiterals) {
        if (Modifier.isStatic(field.getModifiers())) {
          errorMessenger.addError("Expected to be non-static for field (%s)", field);
        }

        if (field.getType() != Class.class) {
          errorMessenger.addError("Expected a class literal type (Class<?>) for field (%s)", field);
        }

        LoadClass loadClassAnnotation = field.getDeclaredAnnotation(LoadClass.class);
        int round = loadClassAnnotation.round();
        if (round < 0 || round > maxNumOfTransformations) {
          errorMessenger.addError(
              "Expected the round (Actual:%d) of desugar transformation within [0, %d], where 0"
                  + " indicates no transformation is applied.",
              round, maxNumOfTransformations);
        }
      }
    }

    private void checkInjectableAsmNodes() {
      for (Field field : injectableAsmNodes) {
        if (Modifier.isStatic(field.getModifiers())) {
          errorMessenger.addError("Expected to be non-static for field (%s)", field);
        }

        if (!SUPPORTED_ASM_NODE_TYPES.contains(field.getType())) {
          errorMessenger.addError(
              "Expected @LoadAsmNode on a field in one of: (%s) but gets (%s)",
              SUPPORTED_ASM_NODE_TYPES, field);
        }

        LoadAsmNode astAsmNodeInfo = field.getDeclaredAnnotation(LoadAsmNode.class);
        int round = astAsmNodeInfo.round();
        if (round < 0 || round > maxNumOfTransformations) {
          errorMessenger.addError(
              "Expected the round (actual:%d) of desugar transformation within [0, %d], where 0"
                  + " indicates no transformation is used.",
              round, maxNumOfTransformations);
        }
      }
    }

    private void checkInjectableZipEntries() {
      for (Field field : injectableJarFileEntries) {
        if (Modifier.isStatic(field.getModifiers())) {
          errorMessenger.addError("Expected to be non-static for field (%s)", field);
        }

        if (field.getType() != ZipEntry.class) {
          errorMessenger.addError(
              "Expected a field with Type: (%s) but gets (%s)", ZipEntry.class.getName(), field);
        }

        LoadZipEntry zipEntryInfo = field.getDeclaredAnnotation(LoadZipEntry.class);
        if (zipEntryInfo.round() < 0 || zipEntryInfo.round() > maxNumOfTransformations) {
          errorMessenger.addError(
              "Expected the round of desugar transformation within [0, %d], where 0 indicates no"
                  + " transformation is used.",
              maxNumOfTransformations);
        }
      }
    }
  }

  /** A messenger that manages desugar configuration errors. */
  private static class ErrorMessenger {

    private final List<String> errorMessages = new ArrayList<>();

    @FormatMethod
    ErrorMessenger addError(String recipe, Object... args) {
      errorMessages.add(String.format(recipe, args));
      return this;
    }

    boolean containsAnyError() {
      return !errorMessages.isEmpty();
    }

    List<String> getAllMessages() {
      return errorMessages;
    }

    @Override
    public String toString() {
      return getAllMessages().toString();
    }
  }
}
