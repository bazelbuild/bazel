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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Strings;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import com.google.devtools.build.android.desugar.Desugar;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberKey;
import com.google.devtools.build.android.desugar.langmodel.FieldKey;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle.MemberUseContext;
import java.io.IOException;
import java.io.InputStream;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.runner.RunWith;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;

/** The builder class for {@link DesugarRule}. */
public class DesugarRuleBuilder {

  private static final String DEFAULT_OUTPUT_ROOT_PREFIX = "desugared_dump";

  private List<JarTransformationRecord> jarTransformationRecords;

  /** The state of the already-created directories to avoid directory re-creation. */
  private final Map<String, Path> tempDirs = new HashMap<>();

  /** A table for the lookup of missing user-supplied class member descriptors. */
  private final Table<
          Integer, // Desugar round
          ClassMemberKey, // A class member without descriptor (empty descriptor string).
          Set<ClassMemberKey>> // The set of same-name class members with their descriptors.
      descriptorLookupRepo = HashBasedTable.create();

  /**
   * A table for the lookup of reflection-based class member representation from round and class
   * member key.
   */
  private final Table<
          Integer, // Desugar round
          ClassMemberKey, // A class member with descriptor.
          java.lang.reflect.Member> // A reflection-based Member instance.
      reflectionBasedMembers = HashBasedTable.create();

  private static final ImmutableSet<Class<?>> SUPPORTED_ASM_NODE_TYPES =
      ImmutableSet.of(ClassNode.class, FieldNode.class, MethodNode.class);

  private final Object testInstance;
  private final MethodHandles.Lookup testInstanceLookup;
  private final ImmutableList<Field> injectableClassLiterals;
  private final ImmutableList<Field> injectableAsmNodes;
  private final ImmutableList<Field> injectableMethodHandles;
  private final ImmutableList<Field> injectableZipFileEntries;
  private String workingJavaPackage = "";
  private int maxNumOfTransformations = 1;
  private final List<Path> inputs = new ArrayList<>();
  private final List<Path> classPathEntries = new ArrayList<>();
  private final List<Path> bootClassPathEntries = new ArrayList<>();
  private final Multimap<String, String> customCommandOptions = LinkedHashMultimap.create();
  private final ErrorMessenger errorMessenger = new ErrorMessenger();

  private final Path androidRuntimeJar;
  private final Path jacocoAgentJar;

  private final Map<DynamicClassLiteral, Class<?>> dynamicClassLiterals = new HashMap<>();
  private final Map<AsmNode, Object> asmNodes = new HashMap<>();
  private final Map<RuntimeMethodHandle, MethodHandle> runtimeMethodHandles = new HashMap<>();
  private final Map<RuntimeZipEntry, ZipEntry> runtimeZipEntries = new HashMap<>();

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
    injectableZipFileEntries =
        DesugarRule.findAllInjectableFieldsWithQualifier(testClass, RuntimeZipEntry.class);
  }

  private void resolve() throws Throwable {
    jarTransformationRecords = new ArrayList<>(maxNumOfTransformations);
    ImmutableList<Path> transInputs = ImmutableList.copyOf(inputs);
    for (int round = 1; round <= maxNumOfTransformations; round++) {
      ImmutableList<Path> transOutputs =
          getRuntimeOutputPaths(
              transInputs,
              tempDirs,
              /* outputRootPrefix= */ DEFAULT_OUTPUT_ROOT_PREFIX + "_" + round);
      JarTransformationRecord transformationRecord =
          JarTransformationRecord.create(
              transInputs,
              transOutputs,
              ImmutableList.copyOf(classPathEntries),
              ImmutableList.copyOf(bootClassPathEntries),
              ImmutableListMultimap.copyOf(customCommandOptions));
      Desugar.main(transformationRecord.getDesugarFlags().toArray(new String[0]));

      jarTransformationRecords.add(transformationRecord);
      transInputs = transOutputs;
    }

    ClassLoader inputClassLoader = getInputClassLoader();
    for (Field field : injectableClassLiterals) {
      DynamicClassLiteral classLiteralRequest =
          field.getDeclaredAnnotation(DynamicClassLiteral.class);
      Class<?> classLiteral =
          loadClassLiteral(
              classLiteralRequest,
              jarTransformationRecords,
              inputClassLoader,
              reflectionBasedMembers,
              descriptorLookupRepo,
              workingJavaPackage);
      dynamicClassLiterals.put(classLiteralRequest, classLiteral);
    }

    for (Field field : injectableAsmNodes) {
      Class<?> requestedFieldType = field.getType();
      AsmNode asmNodeRequest = field.getDeclaredAnnotation(AsmNode.class);
      Object asmNode =
          getAsmNode(
              asmNodeRequest,
              requestedFieldType,
              jarTransformationRecords,
              ImmutableList.copyOf(inputs),
              workingJavaPackage);
      asmNodes.put(asmNodeRequest, asmNode);
    }

    for (Field field : injectableMethodHandles) {
      RuntimeMethodHandle runtimeMethodHandleRequest =
          field.getDeclaredAnnotation(RuntimeMethodHandle.class);
      MethodHandle methodHandle =
          getMethodHandle(
              runtimeMethodHandleRequest,
              testInstanceLookup,
              jarTransformationRecords,
              inputClassLoader,
              reflectionBasedMembers,
              descriptorLookupRepo,
              workingJavaPackage);
      runtimeMethodHandles.put(runtimeMethodHandleRequest, methodHandle);
    }

    for (Field field : injectableZipFileEntries) {
      RuntimeZipEntry runtimeZipEntry = field.getDeclaredAnnotation(RuntimeZipEntry.class);
      ZipEntry zipEntry =
          getZipEntry(
              runtimeZipEntry,
              jarTransformationRecords,
              ImmutableList.copyOf(inputs),
              workingJavaPackage);
      runtimeZipEntries.put(runtimeZipEntry, zipEntry);
    }
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

    if (bootClassPathEntries.isEmpty()
        && !customCommandOptions.containsKey("allow_empty_bootclasspath")) {
      addCommandOptions("bootclasspath_entry", androidRuntimeJar.toString());
    }

    addClasspathEntries(jacocoAgentJar);

    try {
      resolve();
    } catch (Throwable throwable) {
      errorMessenger.addError("Failed to resolve dependencies: %s", throwable);
    }

    if (errorMessenger.containsAnyError()) {
      throw new IllegalStateException(
          String.format(
              "Invalid Desugar configurations:\n%s\n",
              String.join("\n", errorMessenger.getAllMessages())));
    }

    return new DesugarRule(
        injectableAsmNodes,
        testInstance,
        testInstanceLookup,
        injectableClassLiterals,
        injectableMethodHandles,
        injectableZipFileEntries,
        androidRuntimeJar,
        jacocoAgentJar,
        dynamicClassLiterals,
        asmNodes,
        runtimeMethodHandles,
        runtimeZipEntries);
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
    for (Field field : injectableZipFileEntries) {
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

  private static void fillMissingClassMemberDescriptorRepo(
      int round,
      Class<?> classLiteral,
      Table<Integer, ClassMemberKey, Set<ClassMemberKey>> missingDescriptorLookupRepo) {
    String ownerName = Type.getInternalName(classLiteral);
    for (Constructor<?> constructor : classLiteral.getDeclaredConstructors()) {
      ClassMemberKey memberKeyWithoutDescriptor = MethodKey.create(ownerName, "<init>", "");
      ClassMemberKey memberKeyWithDescriptor =
          MethodKey.create(ownerName, "<init>", Type.getConstructorDescriptor(constructor));
      if (missingDescriptorLookupRepo.contains(round, memberKeyWithoutDescriptor)) {
        missingDescriptorLookupRepo
            .get(round, memberKeyWithoutDescriptor)
            .add(memberKeyWithDescriptor);
      } else {
        missingDescriptorLookupRepo.put(
            round, memberKeyWithoutDescriptor, Sets.newHashSet(memberKeyWithDescriptor));
      }
    }
    for (Method method : classLiteral.getDeclaredMethods()) {
      ClassMemberKey memberKeyWithoutDescriptor = MethodKey.create(ownerName, method.getName(), "");
      ClassMemberKey memberKeyWithDescriptor =
          MethodKey.create(ownerName, method.getName(), Type.getMethodDescriptor(method));
      if (missingDescriptorLookupRepo.contains(round, memberKeyWithoutDescriptor)) {
        missingDescriptorLookupRepo
            .get(round, memberKeyWithoutDescriptor)
            .add(memberKeyWithDescriptor);
      } else {
        missingDescriptorLookupRepo.put(
            round, memberKeyWithoutDescriptor, Sets.newHashSet(memberKeyWithDescriptor));
      }
    }
    for (Field field : classLiteral.getDeclaredFields()) {
      ClassMemberKey memberKeyWithoutDescriptor = FieldKey.create(ownerName, field.getName(), "");
      ClassMemberKey memberKeyWithDescriptor =
          FieldKey.create(ownerName, field.getName(), Type.getDescriptor(field.getType()));
      if (missingDescriptorLookupRepo.contains(round, memberKeyWithoutDescriptor)) {
        missingDescriptorLookupRepo
            .get(round, memberKeyWithoutDescriptor)
            .add(memberKeyWithDescriptor);
      } else {
        missingDescriptorLookupRepo.put(
            round, memberKeyWithoutDescriptor, Sets.newHashSet(memberKeyWithDescriptor));
      }
    }
  }

  private static ImmutableTable<Integer, ClassMemberKey, Member> getReflectionBasedClassMembers(
      int round, Class<?> classLiteral) {
    ImmutableTable.Builder<Integer, ClassMemberKey, Member> reflectionBasedMembers =
        ImmutableTable.builder();
    String ownerName = Type.getInternalName(classLiteral);
    for (Field field : classLiteral.getDeclaredFields()) {
      reflectionBasedMembers.put(
          round,
          FieldKey.create(ownerName, field.getName(), Type.getDescriptor(field.getType())),
          field);
    }
    for (Constructor<?> constructor : classLiteral.getDeclaredConstructors()) {
      reflectionBasedMembers.put(
          round,
          MethodKey.create(ownerName, "<init>", Type.getConstructorDescriptor(constructor)),
          constructor);
    }
    for (Method method : classLiteral.getDeclaredMethods()) {
      reflectionBasedMembers.put(
          round,
          MethodKey.create(ownerName, method.getName(), Type.getMethodDescriptor(method)),
          method);
    }
    return reflectionBasedMembers.build();
  }

  private static Class<?> loadClassLiteral(
      DynamicClassLiteral dynamicClassLiteralRequest,
      List<JarTransformationRecord> jarTransformationRecords,
      ClassLoader initialInputClassLoader,
      Table<Integer, ClassMemberKey, Member> reflectionBasedMembers,
      Table<Integer, ClassMemberKey, Set<ClassMemberKey>> missingDescriptorLookupRepo,
      String workingJavaPackage)
      throws Throwable {
    int round = dynamicClassLiteralRequest.round();
    ClassLoader outputJarClassLoader =
        round == 0
            ? initialInputClassLoader
            : jarTransformationRecords.get(round - 1).getOutputClassLoader();
    String requestedClassName = dynamicClassLiteralRequest.value();
    String qualifiedClassName =
        workingJavaPackage.isEmpty() || requestedClassName.contains(".")
            ? requestedClassName
            : workingJavaPackage + "." + requestedClassName;
    Class<?> classLiteral = outputJarClassLoader.loadClass(qualifiedClassName);
    reflectionBasedMembers.putAll(getReflectionBasedClassMembers(round, classLiteral));
    fillMissingClassMemberDescriptorRepo(round, classLiteral, missingDescriptorLookupRepo);
    return classLiteral;
  }

  private static <T> T getAsmNode(
      AsmNode asmNodeRequest,
      Class<T> requestedNodeType,
      List<JarTransformationRecord> jarTransformationRecords,
      ImmutableList<Path> initialInputs,
      String workingJavaPackage)
      throws IOException, ClassNotFoundException {
    String requestedClassName = asmNodeRequest.className();
    String qualifiedClassName =
        workingJavaPackage.isEmpty() || requestedClassName.contains(".")
            ? requestedClassName
            : workingJavaPackage + "." + requestedClassName;
    String classFileName = qualifiedClassName.replace('.', '/') + ".class";
    int round = asmNodeRequest.round();
    ImmutableList<Path> jars =
        round == 0 ? initialInputs : jarTransformationRecords.get(round - 1).outputJars();
    ClassNode classNode = findClassNode(classFileName, jars);
    if (requestedNodeType == ClassNode.class) {
      return requestedNodeType.cast(classNode);
    }

    String memberName = asmNodeRequest.memberName();
    String memberDescriptor = asmNodeRequest.memberDescriptor();
    if (requestedNodeType == FieldNode.class) {
      return requestedNodeType.cast(getFieldNode(classNode, memberName, memberDescriptor));
    }
    if (requestedNodeType == MethodNode.class) {
      return requestedNodeType.cast(getMethodNode(classNode, memberName, memberDescriptor));
    }

    throw new UnsupportedOperationException(
        String.format("Injecting a node type (%s) is not supported", requestedNodeType));
  }

  private static MethodHandle getMethodHandle(
      RuntimeMethodHandle methodHandleRequest,
      Lookup lookup,
      List<JarTransformationRecord> jarTransformationRecords,
      ClassLoader initialInputClassLoader,
      Table<Integer, ClassMemberKey, java.lang.reflect.Member> reflectionBasedMembers,
      Table<Integer, ClassMemberKey, Set<ClassMemberKey>> missingDescriptorLookupRepo,
      String workingJavaPackage)
      throws Throwable {
    int round = methodHandleRequest.round();
    Class<?> classLiteral =
        loadClassLiteral(
            DesugarRule.createDynamicClassLiteral(methodHandleRequest.className(), round),
            jarTransformationRecords,
            initialInputClassLoader,
            reflectionBasedMembers,
            missingDescriptorLookupRepo,
            workingJavaPackage);

    String ownerInternalName = Type.getInternalName(classLiteral);
    String memberName = methodHandleRequest.memberName();
    String memberDescriptor = methodHandleRequest.memberDescriptor();

    switch (methodHandleRequest.usage()) {
      case METHOD_INVOCATION:
        MethodKey methodKey = MethodKey.create(ownerInternalName, memberName, memberDescriptor);
        if (methodKey.descriptor().isEmpty()) {
          Set<ClassMemberKey> memberKeys = missingDescriptorLookupRepo.get(round, methodKey);
          if (memberKeys.size() > 1) {
            throw new IllegalStateException(
                String.format(
                    "Method (%s) has same-name overloaded methods: (%s) \n"
                        + "Please specify a descriptor to disambiguate overloaded method request.",
                    methodKey, memberKeys));
          }

          methodKey = (MethodKey) Iterables.getOnlyElement(memberKeys);
        }
        if (methodKey.isConstructor()) {
          return lookup.unreflectConstructor(
              (Constructor<?>) reflectionBasedMembers.get(round, methodKey));
        } else {
          return lookup.unreflect((Method) reflectionBasedMembers.get(round, methodKey));
        }
      case FIELD_GETTER:
        {
          FieldKey fieldKey = FieldKey.create(ownerInternalName, memberName, memberDescriptor);
          if (fieldKey.descriptor().isEmpty()) {
            Set<ClassMemberKey> memberKeys = missingDescriptorLookupRepo.get(round, fieldKey);
            fieldKey = (FieldKey) Iterables.getOnlyElement(memberKeys);
          }

          return lookup.unreflectGetter((Field) reflectionBasedMembers.get(round, fieldKey));
        }
      case FIELD_SETTER:
        {
          FieldKey fieldKey = FieldKey.create(ownerInternalName, memberName, memberDescriptor);
          if (fieldKey.descriptor().isEmpty()) {
            Set<ClassMemberKey> memberKeys = missingDescriptorLookupRepo.get(round, fieldKey);
            fieldKey = (FieldKey) Iterables.getOnlyElement(memberKeys);
          }
          return lookup.unreflectSetter((Field) reflectionBasedMembers.get(round, fieldKey));
        }
    }
    throw new AssertionError(
        String.format(
            "Beyond exhaustive enum values: Unexpected enum value (%s) for (Enum:%s)",
            methodHandleRequest.usage(), MemberUseContext.class));
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

  private static ZipEntry getZipEntry(
      RuntimeZipEntry zipEntryRequest,
      List<JarTransformationRecord> jarTransformationRecords,
      ImmutableList<Path> initialInputs,
      String workingJavaPackage)
      throws IOException {
    String requestedClassFile = zipEntryRequest.value();
    String zipEntryPathName =
        workingJavaPackage.isEmpty() || requestedClassFile.contains("/")
            ? requestedClassFile
            : workingJavaPackage.replace('.', '/') + '/' + requestedClassFile;
    int round = zipEntryRequest.round();
    ImmutableList<Path> jars =
        round == 0 ? initialInputs : jarTransformationRecords.get(round - 1).outputJars();
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
    return URLClassLoader.newInstance(urls.toArray(new URL[0]), DesugarRule.BASE_CLASS_LOADER);
  }

  private static ImmutableList<Path> getRuntimeOutputPaths(
      ImmutableList<Path> inputs, Map<String, Path> tempDirs, String outputRootPrefix)
      throws IOException {
    ImmutableList.Builder<Path> outputRuntimePathsBuilder = ImmutableList.builder();
    for (Path path : inputs) {
      String targetDirKey = Paths.get(outputRootPrefix) + "/" + path.getParent();
      final Path outputDirPath;
      if (tempDirs.containsKey(targetDirKey)) {
        outputDirPath = tempDirs.get(targetDirKey);
      } else {
        Path root = Files.createTempDirectory("junit");
        Files.delete(root);
        outputDirPath = Files.createDirectories(root.resolve(Paths.get(targetDirKey)));
        tempDirs.put(targetDirKey, outputDirPath);
      }
      outputRuntimePathsBuilder.add(outputDirPath.resolve(path.getFileName()));
    }
    return outputRuntimePathsBuilder.build();
  }
}
