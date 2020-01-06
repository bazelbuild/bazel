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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.zip.ZipEntry;
import org.junit.runner.RunWith;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;

/** The builder class for {@link DesugarRule}. */
public class DesugarRuleBuilder {

  private static final ImmutableSet<Class<?>> SUPPORTED_ASM_NODE_TYPES =
      ImmutableSet.of(ClassNode.class, FieldNode.class, MethodNode.class);

  private final Object testInstance;
  private final MethodHandles.Lookup testInstanceLookup;
  private final ImmutableList<Field> injectableClassLiterals;
  private final ImmutableList<Field> injectableAsmNodes;
  private final ImmutableList<Field> injectableMethodHandles;
  private final ImmutableList<Field> injectableJarFileEntries;
  private String workingJavaPackage = "";
  private int maxNumOfTransformations = 1;
  private final List<Path> inputs = new ArrayList<>();
  private final List<Path> classPathEntries = new ArrayList<>();
  private final List<Path> bootClassPathEntries = new ArrayList<>();
  private final Multimap<String, String> customCommandOptions = LinkedHashMultimap.create();
  private final ErrorMessenger errorMessenger = new ErrorMessenger();

  private final Path androidRuntimeJar;
  private final Path jacocoAgentJar;

  DesugarRuleBuilder(Object testInstance, MethodHandles.Lookup testInstanceLookup) {
    this.testInstance = testInstance;
    this.testInstanceLookup = testInstanceLookup;
    Class<?> testClass = testInstance.getClass();

    androidRuntimeJar = Paths.get(getExplicitJvmFlagValue("android_runtime_jar", errorMessenger));
    jacocoAgentJar = Paths.get(getExplicitJvmFlagValue("jacoco_agent_jar", errorMessenger));

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
        DesugarRule.findAllInjectableFieldsWithQualifier(testClass, DynamicClassLiteral.class);
    injectableAsmNodes = DesugarRule.findAllInjectableFieldsWithQualifier(testClass, AsmNode.class);
    injectableMethodHandles =
        DesugarRule.findAllInjectableFieldsWithQualifier(testClass, RuntimeMethodHandle.class);
    injectableJarFileEntries =
        DesugarRule.findAllInjectableFieldsWithQualifier(testClass, RuntimeZipEntry.class);
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
    checkInjectableMethodHandles();
    checkInjectableZipEntries();

    if (!Files.exists(androidRuntimeJar)) {
      errorMessenger.addError("Android Runtime Jar does not exist: %s", androidRuntimeJar);
    }

    if (!Files.exists(jacocoAgentJar)) {
      errorMessenger.addError("Jacoco Agent Jar does not exist: %s", jacocoAgentJar);
    }

    if (errorMessenger.containsAnyError()) {
      throw new IllegalStateException(
          String.format(
              "Invalid Desugar configurations:\n%s\n",
              String.join("\n", errorMessenger.getAllMessages())));
    }

    if (bootClassPathEntries.isEmpty()
        && !customCommandOptions.containsKey("allow_empty_bootclasspath")) {
      addCommandOptions("bootclasspath_entry", androidRuntimeJar.toString());
    }

    addClasspathEntries(jacocoAgentJar);

    return new DesugarRule(
        injectableJarFileEntries,
        testInstance,
        testInstanceLookup,
        maxNumOfTransformations,
        workingJavaPackage,
        new ArrayList<>(maxNumOfTransformations),
        ImmutableList.copyOf(bootClassPathEntries),
        ImmutableListMultimap.copyOf(customCommandOptions),
        ImmutableList.copyOf(inputs),
        ImmutableList.copyOf(classPathEntries),
        injectableClassLiterals,
        injectableAsmNodes,
        injectableMethodHandles,
        androidRuntimeJar,
        jacocoAgentJar);
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

  private void checkInjectableZipEntries() {
    for (Field field : injectableJarFileEntries) {
      if (Modifier.isStatic(field.getModifiers())) {
        errorMessenger.addError("Expected to be non-static for field (%s)", field);
      }

      if (field.getType() != ZipEntry.class) {
        errorMessenger.addError(
            "Expected a field with Type: (%s) but gets (%s)", ZipEntry.class.getName(), field);
      }

      RuntimeZipEntry zipEntryInfo = field.getDeclaredAnnotation(RuntimeZipEntry.class);
      if (zipEntryInfo.round() < 0 || zipEntryInfo.round() > maxNumOfTransformations) {
        errorMessenger.addError(
            "Expected the round of desugar transformation within [0, %d], where 0 indicates no"
                + " transformation is used.",
            maxNumOfTransformations);
      }
    }
  }

  private static String getExplicitJvmFlagValue(String jvmFlagKey, ErrorMessenger errorMessenger) {
    String jvmFlagValue = System.getProperty(jvmFlagKey);
    if (Strings.isNullOrEmpty(jvmFlagValue)) {
      errorMessenger.addError(
          "Expected JVM flag specified: -D%s=<value of %s>", jvmFlagKey, jvmFlagKey);
      return "";
    }
    return jvmFlagValue;
  }
}
