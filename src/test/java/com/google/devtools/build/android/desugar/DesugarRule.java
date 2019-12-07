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

import com.google.auto.value.AutoValue;
import com.google.common.annotations.UsedReflectively;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.runtime.RunfilesPaths;
import com.google.errorprone.annotations.FormatMethod;
import java.io.IOException;
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
import org.junit.rules.TemporaryFolder;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.model.Statement;

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
   *           .addRuntimeInputs("MyJar.jar")
   *           .build();
   *
   *   &#064;LoadClass("my.package.ClassToDesugar")
   *   private Class<?> classToDesugarClass;f
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

  private static final Path ANDROID_RUNTIME_JAR_PATH =
      RunfilesPaths.resolve(
          "third_party/java/android/android_sdk_linux/platforms/experimental/android_blaze.jar");

  private static final Path JACOCO_RUNTIME_PATH =
      RunfilesPaths.resolve("third_party/java/jacoco/jacoco_agent.jar");

  private static final String DEFAULT_OUTPUT_ROOT_PREFIX = "desugared_dump";

  private static final ClassLoader baseClassLoader = ClassLoader.getSystemClassLoader().getParent();

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

    final ClassLoader getOutputClassLoader() throws MalformedURLException {
      List<URL> urls = new ArrayList<>();
      for (Path path : Iterables.concat(outputJars(), classPathEntries(), bootClassPathEntries())) {
        urls.add(path.toUri().toURL());
      }
      return URLClassLoader.newInstance(urls.toArray(new URL[0]), baseClassLoader);
    }
  }

  /** For hosting desugared jar temporarily. */
  private final TemporaryFolder temporaryFolder = new TemporaryFolder();

  private final Object testInstance;
  private final MethodHandles.Lookup testInstanceLookup;
  private final int maxNumOfTransformations;
  private final ImmutableList<Path> inputs;
  private final ImmutableList<Path> classPathEntries;
  private final ImmutableList<Path> bootClassPathEntries;
  private final ImmutableList<Field> fieldForDynamicClassLoading;
  private final ImmutableListMultimap<String, String> extraCustomCommandOptions;

  private final List<JarTransformationRecord> jarTransformationRecords;
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
      DesugarRuleBuilder desugarRuleBuilder,
      Lookup testInstanceLookup,
      int maxNumOfTransformations,
      ImmutableList<Path> inputJars,
      ImmutableList<Path> classPathEntries,
      ImmutableList<Path> bootClassPathEntries) {
    this.testInstance = desugarRuleBuilder.testInstance;
    this.testInstanceLookup = testInstanceLookup;
    this.maxNumOfTransformations = maxNumOfTransformations;
    this.inputs = inputJars;
    this.classPathEntries = classPathEntries;
    this.bootClassPathEntries = bootClassPathEntries;
    this.extraCustomCommandOptions =
        ImmutableListMultimap.copyOf(desugarRuleBuilder.customCommandOptions);
    this.fieldForDynamicClassLoading =
        findAllFieldsWithAnnotation(testInstance.getClass(), LoadClass.class);
    this.jarTransformationRecords = new ArrayList<>(maxNumOfTransformations);
  }

  @Override
  public Statement apply(Statement base, Description description) {
    return temporaryFolder.apply(
        new Statement() {
          @Override
          public void evaluate() throws Throwable {
            ImmutableList<Path> transInputs = inputs;
            for (int i = 0; i < maxNumOfTransformations; i++) {
              ImmutableList<Path> transOutputs =
                  getRuntimeOutputPaths(
                      transInputs,
                      temporaryFolder,
                      tempDirs,
                      /* defaultOutputRootPrefix= */ DEFAULT_OUTPUT_ROOT_PREFIX + "_" + i);
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

            for (Field field : fieldForDynamicClassLoading) {
              LoadClass loadClassAnnotation = field.getDeclaredAnnotation(LoadClass.class);
              String qualifiedClassName = loadClassAnnotation.value();
              int round = loadClassAnnotation.round();
              ClassLoader outputJarClassLoader =
                  round == 0
                      ? getInputClassLoader()
                      : jarTransformationRecords.get(round - 1).getOutputClassLoader();
              Class<?> classLiteral = outputJarClassLoader.loadClass(qualifiedClassName);
              MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
              fieldSetter.invoke(testInstance, classLiteral);
            }
            base.evaluate();
          }
        },
        description);
  }

  private ClassLoader getInputClassLoader() throws MalformedURLException {
    List<URL> urls = new ArrayList<>();
    for (Path path : Iterables.concat(inputs, classPathEntries, bootClassPathEntries)) {
      urls.add(path.toUri().toURL());
    }
    return URLClassLoader.newInstance(urls.toArray(new URL[0]), baseClassLoader);
  }

  private static ImmutableList<Path> getRuntimeOutputPaths(
      ImmutableList<Path> inputs,
      TemporaryFolder temporaryFolder,
      Map<String, Path> tempDirs,
      String defaultOutputRootPrefix)
      throws IOException {
    ImmutableList.Builder<Path> outputRuntimePathsBuilder = ImmutableList.builder();
    for (Path path : inputs) {
      String targetDirKey = Paths.get(defaultOutputRootPrefix) + "/" + path.getParent();
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
      for (Field field : testClass.getDeclaredFields()) {
        if (field.isAnnotationPresent(annotationClass)) {
          fields.add(field);
        }
      }
    }
    return fields.build();
  }

  /** The builder class for {@link DesugarRule}. */
  public static class DesugarRuleBuilder {

    private final Object testInstance;
    private final MethodHandles.Lookup testInstanceLookup;
    private final ImmutableList<Field> classLiteralFieldToBeLoaded;
    private int maxNumOfTransformations;
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

      classLiteralFieldToBeLoaded = findAllFieldsWithAnnotation(testClass, LoadClass.class);
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
      checkClassLiteralFieldToBeLoaded();

      if (bootClassPathEntries.isEmpty()
          && !customCommandOptions.containsKey("allow_empty_bootclasspath")) {
        addCommandOptions("bootclasspath_entry", ANDROID_RUNTIME_JAR_PATH.toString());
      }

      if (errorMessenger.anyError()) {
        throw new IllegalStateException(
            String.format(
                "Invalid Desugar configurations:\n%s\n", errorMessenger.getAllMessages()));
      }

      addClasspathEntries(JACOCO_RUNTIME_PATH);

      return new DesugarRule(
          this,
          testInstanceLookup,
          maxNumOfTransformations,
          ImmutableList.copyOf(inputs),
          ImmutableList.copyOf(classPathEntries),
          ImmutableList.copyOf(bootClassPathEntries));
    }

    private void checkClassLiteralFieldToBeLoaded() {
      for (Field field : classLiteralFieldToBeLoaded) {
        if (Modifier.isStatic(field.getModifiers())) {
          errorMessenger.addError("Expected to be non-static for field (%s)", field);
        }

        if (field.getType() != Class.class) {
          errorMessenger.addError("Expected a class literal type (Class<?>) for field (%s)", field);
        }

        LoadClass loadClassAnnotation = field.getDeclaredAnnotation(LoadClass.class);
        if (loadClassAnnotation.round() < 0
            || loadClassAnnotation.round() > maxNumOfTransformations) {
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

    boolean anyError() {
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
