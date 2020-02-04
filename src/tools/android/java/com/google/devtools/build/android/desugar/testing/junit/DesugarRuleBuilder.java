/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.testing.junit;

import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.jar.JarEntry;
import javax.inject.Inject;
import javax.tools.ToolProvider;
import org.junit.runner.RunWith;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;

/** The builder class for {@link DesugarRule}. */
public class DesugarRuleBuilder {

  private static final ImmutableSet<Class<?>> SUPPORTED_ASM_NODE_TYPES =
      ImmutableSet.of(ClassNode.class, FieldNode.class, MethodNode.class);

  private static final String ANDROID_RUNTIME_JAR_JVM_FLAG_KEY = "android_runtime_jar";
  private static final String JACOCO_AGENT_JAR_JVM_FLAG_KEY = "jacoco_agent_jar";
  private static final String DUMP_PROXY_CLASSES_JVM_FLAG_KEY =
      "jdk.internal.lambda.dumpProxyClasses";

  private final Object testInstance;
  private final MethodHandles.Lookup testInstanceLookup;
  private final ImmutableList<Field> injectableClassLiterals;
  private final ImmutableList<Field> injectableAsmNodes;
  private final ImmutableList<Field> injectableMethodHandles;
  private final ImmutableList<Field> injectableJarEntries;
  private String workingJavaPackage = "";
  private int maxNumOfTransformations = 1;
  private final List<Path> inputs = new ArrayList<>();
  private final List<Path> sourceInputs = new ArrayList<>();
  private final List<String> javacOptions = new ArrayList<>();
  private final List<Path> classPathEntries = new ArrayList<>();
  private final List<Path> bootClassPathEntries = new ArrayList<>();
  private final Multimap<String, String> customCommandOptions = LinkedHashMultimap.create();
  private final ErrorMessenger errorMessenger = new ErrorMessenger();

  private final String androidRuntimeJarJvmFlagValue;
  private final String jacocoAgentJarJvmFlagValue;

  DesugarRuleBuilder(Object testInstance, MethodHandles.Lookup testInstanceLookup) {
    this.testInstance = testInstance;
    this.testInstanceLookup = testInstanceLookup;
    Class<?> testClass = testInstance.getClass();

    androidRuntimeJarJvmFlagValue = getExplicitJvmFlagValue(ANDROID_RUNTIME_JAR_JVM_FLAG_KEY);
    jacocoAgentJarJvmFlagValue = getExplicitJvmFlagValue(JACOCO_AGENT_JAR_JVM_FLAG_KEY);

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

    injectableClassLiterals =
        findAllInjectableFieldsWithQualifier(testClass, DynamicClassLiteral.class);
    injectableAsmNodes = findAllInjectableFieldsWithQualifier(testClass, AsmNode.class);
    injectableMethodHandles =
        findAllInjectableFieldsWithQualifier(testClass, RuntimeMethodHandle.class);
    injectableJarEntries = findAllInjectableFieldsWithQualifier(testClass, RuntimeJarEntry.class);
  }

  static ImmutableList<Field> findAllInjectableFieldsWithQualifier(
      Class<?> testClass, Class<? extends Annotation> annotationClass) {
    ImmutableList.Builder<Field> fields = ImmutableList.builder();
    for (Class<?> currentClass = testClass;
        currentClass != null;
        currentClass = currentClass.getSuperclass()) {
      for (Field field : currentClass.getDeclaredFields()) {
        if (field.isAnnotationPresent(Inject.class) && field.isAnnotationPresent(annotationClass)) {
          fields.add(field);
        }
      }
    }
    return fields.build();
  }

  public DesugarRuleBuilder setWorkingJavaPackage(String workingJavaPackage) {
    this.workingJavaPackage = workingJavaPackage;
    return this;
  }

  public DesugarRuleBuilder enableIterativeTransformation(int maxNumOfTransformations) {
    this.maxNumOfTransformations = maxNumOfTransformations;
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

  /** Add Java source files subject to be compiled during test execution. */
  public DesugarRuleBuilder addSourceInputs(Path... inputSourceFiles) {
    for (Path path : inputSourceFiles) {
      if (!path.toString().endsWith(".java")) {
        errorMessenger.addError("Expected a Java source file (*.java): Actual (%s)", path);
      }
      if (!Files.exists(path)) {
        errorMessenger.addError("File does not exist: %s", path);
      }
      sourceInputs.add(path);
    }
    return this;
  }

  /**
   * Add JVM-flag-specified Java source files subject to be compiled during test execution. It is
   * expected the value associated with `jvmFlagKey` to be a space-separated Strings. E.g. on the
   * command line you would set it like: -Dinput_srcs="path1 path2 path3", and use <code>
   *  .addSourceInputsFromJvmFlag("input_srcs").</code> in your test class.
   */
  public DesugarRuleBuilder addSourceInputsFromJvmFlag(String jvmFlagKey) {
    return addSourceInputs(getRuntimePathsFromJvmFlag(jvmFlagKey));
  }

  /**
   * A helper method that reads file paths into an array from the JVM flag value associated with
   * {@param jvmFlagKey}.
   */
  private static Path[] getRuntimePathsFromJvmFlag(String jvmFlagKey) {
    return Splitter.on(" ").trimResults().splitToList(System.getProperty(jvmFlagKey)).stream()
        .map(Paths::get)
        .toArray(Path[]::new);
  }

  /**
   * Add javac options used for compilation, with the same support of `javacopts` attribute in
   * java_binary rule.
   */
  public DesugarRuleBuilder addJavacOptions(String... javacOptions) {
    for (String javacOption : javacOptions) {
      if (!javacOption.startsWith("-")) {
        errorMessenger.addError(
            "Expected javac options, e.g. `-source 11`, `-target 11`, `-parameters`, Run `javac"
                + " -help` from terminal for all supported options.");
      }
      this.javacOptions.add(javacOption);
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
    if (Strings.isNullOrEmpty(getExplicitJvmFlagValue(DUMP_PROXY_CLASSES_JVM_FLAG_KEY))) {
      errorMessenger.addError(
          "Expected JVM flag: \"-D%s=$$(mktemp -d)\", but it is absent.",
          DUMP_PROXY_CLASSES_JVM_FLAG_KEY);
    }

    if (Strings.isNullOrEmpty(androidRuntimeJarJvmFlagValue)
        || !androidRuntimeJarJvmFlagValue.endsWith(".jar")
        || !Files.exists(Paths.get(androidRuntimeJarJvmFlagValue))) {
      errorMessenger.addError(
          "Android Runtime Jar does not exist: Please check JVM flag: -D%s='%s'",
          ANDROID_RUNTIME_JAR_JVM_FLAG_KEY, androidRuntimeJarJvmFlagValue);
    }

    if (Strings.isNullOrEmpty(jacocoAgentJarJvmFlagValue)
        || !jacocoAgentJarJvmFlagValue.endsWith(".jar")
        || !Files.exists(Paths.get(jacocoAgentJarJvmFlagValue))) {
      errorMessenger.addError(
          "Jacoco Agent Jar does not exist: Please check JVM flag: -D%s='%s'",
          JACOCO_AGENT_JAR_JVM_FLAG_KEY, jacocoAgentJarJvmFlagValue);
    }
  }

  public DesugarRule build() {
    checkInjectableClassLiterals();
    checkInjectableAsmNodes();
    checkInjectableMethodHandles();
    checkInjectableJarEntries();

    if (maxNumOfTransformations > 0) {
      checkJVMOptions();
      if (bootClassPathEntries.isEmpty()
          && !customCommandOptions.containsKey("allow_empty_bootclasspath")) {
        addBootClassPathEntries(Paths.get(androidRuntimeJarJvmFlagValue));
      }
      addClasspathEntries(Paths.get(jacocoAgentJarJvmFlagValue));
    }

    ImmutableList.Builder<SourceCompilationUnit> sourceCompilationUnits = ImmutableList.builder();
    if (!sourceInputs.isEmpty()) {
      try {
        Path runtimeCompiledJar = Files.createTempFile("runtime_compiled_", ".jar");
        sourceCompilationUnits.add(
            new SourceCompilationUnit(
                ToolProvider.getSystemJavaCompiler(),
                ImmutableList.copyOf(javacOptions),
                ImmutableList.copyOf(sourceInputs),
                runtimeCompiledJar));
        addInputs(runtimeCompiledJar);
      } catch (IOException e) {
        errorMessenger.addError(
            "Failed to access the output jar location for compilation: %s\n%s", sourceInputs, e);
      }
    }

    RuntimeEntityResolver runtimeEntityResolver =
        new RuntimeEntityResolver(
            testInstanceLookup,
            workingJavaPackage,
            maxNumOfTransformations,
            ImmutableList.copyOf(inputs),
            ImmutableList.copyOf(classPathEntries),
            ImmutableList.copyOf(bootClassPathEntries),
            ImmutableListMultimap.copyOf(customCommandOptions));

    if (errorMessenger.containsAnyError()) {
      throw new IllegalStateException(
          String.format(
              "Invalid Desugar configurations:\n%s\n",
              String.join("\n", errorMessenger.getAllMessages())));
    }

    return new DesugarRule(
        testInstance,
        testInstanceLookup,
        ImmutableList.<Field>builder()
            .addAll(injectableClassLiterals)
            .addAll(injectableAsmNodes)
            .addAll(injectableMethodHandles)
            .addAll(injectableJarEntries)
            .build(),
        sourceCompilationUnits.build(),
        runtimeEntityResolver);
  }

  private void checkInjectableClassLiterals() {
    for (Field field : injectableClassLiterals) {
      if (Modifier.isStatic(field.getModifiers())) {
        errorMessenger.addError("Expected to be non-static for field (%s)", field);
      }

      if (field.getType() != Class.class) {
        errorMessenger.addError("Expected a class literal type (Class<?>) for field (%s)", field);
      }

      DynamicClassLiteral dynamicClassLiteralAnnotation =
          field.getDeclaredAnnotation(DynamicClassLiteral.class);
      int round = dynamicClassLiteralAnnotation.round();
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
            "Expected @inject @AsmNode on a field in one of: (%s) but gets (%s)",
            SUPPORTED_ASM_NODE_TYPES, field);
      }

      AsmNode astAsmNodeInfo = field.getDeclaredAnnotation(AsmNode.class);
      int round = astAsmNodeInfo.round();
      if (round < 0 || round > maxNumOfTransformations) {
        errorMessenger.addError(
            "Expected the round (actual:%d) of desugar transformation within [0, %d], where 0"
                + " indicates no transformation is used.",
            round, maxNumOfTransformations);
      }
    }
  }

  private void checkInjectableMethodHandles() {
    for (Field field : injectableMethodHandles) {
      if (Modifier.isStatic(field.getModifiers())) {
        errorMessenger.addError("Expected to be non-static for field (%s)", field);
      }

      if (field.getType() != MethodHandle.class) {
        errorMessenger.addError(
            "Expected @Inject @RuntimeMethodHandle annotated on a field with type (%s), but gets"
                + " (%s)",
            MethodHandle.class, field);
      }

      RuntimeMethodHandle methodHandleRequest =
          field.getDeclaredAnnotation(RuntimeMethodHandle.class);
      int round = methodHandleRequest.round();
      if (round < 0 || round > maxNumOfTransformations) {
        errorMessenger.addError(
            "Expected the round (actual:%d) of desugar transformation within [0, %d], where 0"
                + " indicates no transformation is used.",
            round, maxNumOfTransformations);
      }
    }
  }

  private void checkInjectableJarEntries() {
    for (Field field : injectableJarEntries) {
      if (Modifier.isStatic(field.getModifiers())) {
        errorMessenger.addError("Expected to be non-static for field (%s)", field);
      }

      if (field.getType() != JarEntry.class) {
        errorMessenger.addError(
            "Expected a field with Type: (%s) but gets (%s)", JarEntry.class.getName(), field);
      }

      RuntimeJarEntry jarEntryInfo = field.getDeclaredAnnotation(RuntimeJarEntry.class);
      if (jarEntryInfo.round() < 0 || jarEntryInfo.round() > maxNumOfTransformations) {
        errorMessenger.addError(
            "Expected the round of desugar transformation within [0, %d], where 0 indicates no"
                + " transformation is used.",
            maxNumOfTransformations);
      }
    }
  }

  private static String getExplicitJvmFlagValue(String jvmFlagKey) {
    String jvmFlagValue = System.getProperty(jvmFlagKey);
    return Strings.isNullOrEmpty(jvmFlagValue) ? "" : jvmFlagValue;
  }
}
