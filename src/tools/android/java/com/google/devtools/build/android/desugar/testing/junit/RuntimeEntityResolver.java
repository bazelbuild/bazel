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
import static org.objectweb.asm.ClassReader.SKIP_CODE;
import static org.objectweb.asm.ClassReader.SKIP_DEBUG;
import static org.objectweb.asm.ClassReader.SKIP_FRAMES;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import com.google.devtools.build.android.desugar.Desugar;
import com.google.devtools.build.android.desugar.io.JarItem;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberKey;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.FieldKey;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle.MemberUseContext;
import java.io.IOException;
import java.io.InputStream;
import java.lang.annotation.Annotation;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.reflect.AnnotatedElement;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;

/** Resolves the dependencies of fields and test method parameters under desugar testing. */
final class RuntimeEntityResolver {

  static final ImmutableSet<Class<? extends Annotation>> SUPPORTED_QUALIFIERS =
      ImmutableSet.of(
          DynamicClassLiteral.class,
          AsmNode.class,
          RuntimeMethodHandle.class,
          RuntimeJarEntry.class);
  private static final String DEFAULT_OUTPUT_ROOT_PREFIX = "desugared_dump";

  private final MethodHandles.Lookup testInstanceLookup;
  private final String workingJavaPackage;
  private final int maxNumOfTransformations;
  private final ImmutableList<Path> inputs;
  private final ImmutableList<Path> classPathEntries;
  private final ImmutableList<Path> bootClassPathEntries;
  private final ImmutableMultimap<String, String> customCommandOptions;

  private final List<JarTransformationRecord> jarTransformationRecords;
  private ClassLoader inputClassLoader;

  /** The state of the already-created directories to avoid directory re-creation. */
  private final Map<String, Path> tempDirs = new HashMap<>();

  /** A table for the lookup of missing user-supplied class member descriptors. */
  private final Table<
          Integer, // Desugar round
          ClassMemberKey<?>, // A class member without descriptor (empty descriptor string).
          Set<ClassMemberKey<?>>> // The set of same-name class members with their descriptors.
      descriptorLookupRepo = HashBasedTable.create();

  /**
   * A table for the lookup of reflection-based class member representation from round and class
   * member key.
   */
  private final Table<
          Integer, // Desugar round
          ClassMemberKey<?>, // A class member with descriptor.
          java.lang.reflect.Member> // A reflection-based Member instance.
      reflectionBasedMembers = HashBasedTable.create();

  RuntimeEntityResolver(
      Lookup testInstanceLookup,
      String workingJavaPackage,
      int maxNumOfTransformations,
      ImmutableList<Path> inputs,
      ImmutableList<Path> classPathEntries,
      ImmutableList<Path> bootClassPathEntries,
      ImmutableMultimap<String, String> customCommandOptions) {
    this.testInstanceLookup = testInstanceLookup;
    this.workingJavaPackage = workingJavaPackage;
    this.maxNumOfTransformations = maxNumOfTransformations;
    this.inputs = inputs;
    this.classPathEntries = classPathEntries;
    this.bootClassPathEntries = bootClassPathEntries;
    this.customCommandOptions = customCommandOptions;
    this.jarTransformationRecords = new ArrayList<>(maxNumOfTransformations);
  }

  void executeTransformation() throws Exception {
    inputClassLoader = getInputClassLoader();
    ImmutableList<Path> transInputs = inputs;
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
  }

  public <T> T resolve(AnnotatedElement element, Class<T> elementType) throws Throwable {
    DynamicClassLiteral dynamicClassLiteralRequest =
        element.getDeclaredAnnotation(DynamicClassLiteral.class);
    if (dynamicClassLiteralRequest != null) {
      return elementType.cast(
          loadClassLiteral(
              dynamicClassLiteralRequest,
              jarTransformationRecords,
              inputClassLoader,
              reflectionBasedMembers,
              descriptorLookupRepo,
              workingJavaPackage));
    }
    AsmNode asmNodeRequest = element.getDeclaredAnnotation(AsmNode.class);
    if (asmNodeRequest != null) {
      return getAsmNode(
          asmNodeRequest, elementType, jarTransformationRecords, inputs, workingJavaPackage);
    }
    RuntimeMethodHandle runtimeMethodHandleRequest =
        element.getDeclaredAnnotation(RuntimeMethodHandle.class);
    if (runtimeMethodHandleRequest != null) {
      return elementType.cast(
          getMethodHandle(
              runtimeMethodHandleRequest,
              testInstanceLookup,
              jarTransformationRecords,
              inputClassLoader,
              reflectionBasedMembers,
              descriptorLookupRepo,
              workingJavaPackage));
    }
    RuntimeJarEntry runtimeJarEntry = element.getDeclaredAnnotation(RuntimeJarEntry.class);
    if (runtimeJarEntry != null) {
      return elementType.cast(
          getJarEntry(runtimeJarEntry, jarTransformationRecords, inputs, workingJavaPackage));
    }
    throw new UnsupportedOperationException(
        "Expected one of the supported types for injection: " + SUPPORTED_QUALIFIERS);
  }

  private static void fillMissingClassMemberDescriptorRepo(
      int round,
      Class<?> classLiteral,
      Table<Integer, ClassMemberKey<?>, Set<ClassMemberKey<?>>> missingDescriptorLookupRepo) {
    ClassName owner = ClassName.create(classLiteral);
    for (Constructor<?> constructor : classLiteral.getDeclaredConstructors()) {
      ClassMemberKey<?> memberKeyWithoutDescriptor = MethodKey.create(owner, "<init>", "");
      ClassMemberKey<?> memberKeyWithDescriptor =
          MethodKey.create(owner, "<init>", Type.getConstructorDescriptor(constructor));
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
      ClassMemberKey<?> memberKeyWithoutDescriptor = MethodKey.create(owner, method.getName(), "");
      ClassMemberKey<?> memberKeyWithDescriptor =
          MethodKey.create(owner, method.getName(), Type.getMethodDescriptor(method));
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
      ClassMemberKey<?> memberKeyWithoutDescriptor = FieldKey.create(owner, field.getName(), "");
      ClassMemberKey<?> memberKeyWithDescriptor =
          FieldKey.create(owner, field.getName(), Type.getDescriptor(field.getType()));
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

  private static ImmutableTable<Integer, ClassMemberKey<?>, Member> getReflectionBasedClassMembers(
      int round, Class<?> classLiteral) {
    ImmutableTable.Builder<Integer, ClassMemberKey<?>, Member> reflectionBasedMembers =
        ImmutableTable.builder();
    ClassName owner = ClassName.create(classLiteral);
    for (Field field : classLiteral.getDeclaredFields()) {
      reflectionBasedMembers.put(
          round,
          FieldKey.create(owner, field.getName(), Type.getDescriptor(field.getType())),
          field);
    }
    for (Constructor<?> constructor : classLiteral.getDeclaredConstructors()) {
      reflectionBasedMembers.put(
          round,
          MethodKey.create(owner, "<init>", Type.getConstructorDescriptor(constructor)),
          constructor);
    }
    for (Method method : classLiteral.getDeclaredMethods()) {
      reflectionBasedMembers.put(
          round,
          MethodKey.create(owner, method.getName(), Type.getMethodDescriptor(method)),
          method);
    }
    return reflectionBasedMembers.build();
  }

  private static Class<?> loadClassLiteral(
      DynamicClassLiteral dynamicClassLiteralRequest,
      List<JarTransformationRecord> jarTransformationRecords,
      ClassLoader initialInputClassLoader,
      Table<Integer, ClassMemberKey<?>, Member> reflectionBasedMembers,
      Table<Integer, ClassMemberKey<?>, Set<ClassMemberKey<?>>> missingDescriptorLookupRepo,
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
      Table<Integer, ClassMemberKey<?>, java.lang.reflect.Member> reflectionBasedMembers,
      Table<Integer, ClassMemberKey<?>, Set<ClassMemberKey<?>>> missingDescriptorLookupRepo,
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

    ClassName owner = ClassName.create(classLiteral);
    String memberName = methodHandleRequest.memberName();
    String memberDescriptor = methodHandleRequest.memberDescriptor();

    ClassMemberKey<?> classMemberKey =
        methodHandleRequest.usage() == MemberUseContext.METHOD_INVOCATION
            ? MethodKey.create(owner, memberName, memberDescriptor)
            : FieldKey.create(owner, memberName, memberDescriptor);

    if (classMemberKey.descriptor().isEmpty()) {
      classMemberKey = restoreMissingDescriptor(classMemberKey, round, missingDescriptorLookupRepo);
    }

    switch (methodHandleRequest.usage()) {
      case METHOD_INVOCATION:
        return classMemberKey.isConstructor()
            ? lookup.unreflectConstructor(
                (Constructor<?>) reflectionBasedMembers.get(round, classMemberKey))
            : lookup.unreflect((Method) reflectionBasedMembers.get(round, classMemberKey));
      case FIELD_GETTER:
        return lookup.unreflectGetter((Field) reflectionBasedMembers.get(round, classMemberKey));
      case FIELD_SETTER:
        return lookup.unreflectSetter((Field) reflectionBasedMembers.get(round, classMemberKey));
    }

    throw new AssertionError(
        String.format(
            "Beyond exhaustive enum values: Unexpected enum value (%s) for (Enum:%s)",
            methodHandleRequest.usage(), MemberUseContext.class));
  }

  private static ClassMemberKey<?> restoreMissingDescriptor(
      ClassMemberKey<?> classMemberKey,
      int round,
      Table<Integer, ClassMemberKey<?>, Set<ClassMemberKey<?>>> missingDescriptorLookupRepo) {
    Set<ClassMemberKey<?>> restoredClassMemberKey =
        missingDescriptorLookupRepo.get(round, classMemberKey);
    if (restoredClassMemberKey == null || restoredClassMemberKey.isEmpty()) {
      throw new IllegalStateException(
          String.format(
              "Unable to find class member (%s). Please check its presence.", classMemberKey));
    } else if (restoredClassMemberKey.size() > 1) {
      throw new IllegalStateException(
          String.format(
              "Class Member (%s) has same-name overloaded members: (%s) \n"
                  + "Please specify a descriptor to disambiguate overloaded method request.",
              classMemberKey, restoredClassMemberKey));
    }
    return Iterables.getOnlyElement(restoredClassMemberKey);
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

  private static JarItem getJarEntry(
      RuntimeJarEntry jarEntryRequest,
      List<JarTransformationRecord> jarTransformationRecords,
      ImmutableList<Path> initialInputs,
      String workingJavaPackage)
      throws IOException {
    String requestedClassFile = jarEntryRequest.value();
    String jarEntryPathName =
        workingJavaPackage.isEmpty() || requestedClassFile.contains("/")
            ? requestedClassFile
            : workingJavaPackage.replace('.', '/') + '/' + requestedClassFile;
    int round = jarEntryRequest.round();
    ImmutableList<Path> jars =
        round == 0 ? initialInputs : jarTransformationRecords.get(round - 1).outputJars();
    for (Path jar : jars) {
      JarFile jarFile = new JarFile(jar.toFile());
      JarEntry jarEntry = jarFile.getJarEntry(jarEntryPathName);
      if (jarEntry != null) {
        return JarItem.create(jarFile, jarEntry);
      }
    }
    throw new IllegalStateException(
        String.format("Expected zip entry of (%s) present.", jarEntryPathName));
  }

  private static ClassNode findClassNode(String jarEntryPathName, ImmutableList<Path> jars)
      throws IOException, ClassNotFoundException {
    for (Path jar : jars) {
      JarFile jarFile = new JarFile(jar.toFile());
      JarEntry jarEntry = jarFile.getJarEntry(jarEntryPathName);
      if (jarEntry != null) {
        try (InputStream inputStream = jarFile.getInputStream(jarEntry)) {
          ClassReader cr = new ClassReader(inputStream);
          ClassNode classNode = new ClassNode(Opcodes.ASM8);
          cr.accept(classNode, 0);
          return classNode;
        }
      }
    }
    throw new ClassNotFoundException(jarEntryPathName);
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

  ImmutableMap<String, Integer> getInputClassFileMajorVersions() throws IOException {
    return getInputClassFileMajorVersions(inputs);
  }

  private static ImmutableMap<String, Integer> getInputClassFileMajorVersions(Collection<Path> jars)
      throws IOException {
    ImmutableMap.Builder<String, Integer> majorVersions = ImmutableMap.builder();
    for (Path jar : jars) {
      JarFile jarFile = new JarFile(jar.toFile());
      List<JarEntry> classFileJarEntries =
          jarFile.stream()
              .filter(jarEntry -> jarEntry.getName().endsWith(".class"))
              .collect(Collectors.toList());
      for (JarEntry jarEntry : classFileJarEntries) {
        try (InputStream inputStream = jarFile.getInputStream(jarEntry)) {
          ClassReader cr = new ClassReader(inputStream);
          ClassNode classNode = new ClassNode(Opcodes.ASM8);
          cr.accept(classNode, SKIP_CODE | SKIP_DEBUG | SKIP_FRAMES);
          majorVersions.put(classNode.name, classNode.version);
        }
      }
    }
    return majorVersions.build();
  }
}
