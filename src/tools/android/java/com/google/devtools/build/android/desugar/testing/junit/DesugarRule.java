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

package com.google.devtools.build.android.desugar.testing.junit;

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoAnnotation;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import com.google.devtools.build.android.desugar.Desugar;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberKey;
import com.google.devtools.build.android.desugar.langmodel.FieldKey;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle.MemberUseContext;
import java.io.IOException;
import java.io.InputStream;
import java.lang.annotation.Annotation;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.inject.Inject;
import org.junit.rules.TemporaryFolder;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;

/** A JUnit4 Rule that desugars an input jar file and load the transformed jar to JVM. */
public final class DesugarRule implements TestRule {

  static final ClassLoader BASE_CLASS_LOADER = ClassLoader.getSystemClassLoader().getParent();

  private static final String DEFAULT_OUTPUT_ROOT_PREFIX = "desugared_dump";

  private final Path androidRuntimeJar;
  private final Path jacocoAgentJar;

  /** For hosting desugared jar temporarily. */
  private final TemporaryFolder temporaryFolder = new TemporaryFolder();

  private final Object testInstance;
  private final MethodHandles.Lookup testInstanceLookup;

  /** The maximum number of desugar operations, used for testing idempotency. */
  private final int maxNumOfTransformations;

  /**
   * The current working java package of desugar operations, when non-empty, used as a prefix to
   * prepend simple class names to get qualified class names.
   */
  private final String workingJavaPackage;

  private final List<JarTransformationRecord> jarTransformationRecords;

  private final ImmutableList<Path> inputs;
  private final ImmutableList<Path> classPathEntries;
  private final ImmutableList<Path> bootClassPathEntries;
  private final ImmutableListMultimap<String, String> extraCustomCommandOptions;

  private final ImmutableList<Field> injectableClassLiterals;
  private final ImmutableList<Field> injectableAsmNodes;
  private final ImmutableList<Field> injectableMethodHandles;
  private final ImmutableList<Field> injectableZipEntries;

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

  DesugarRule(
      ImmutableList<Field> injectableZipEntries,
      Object testInstance,
      Lookup testInstanceLookup,
      int maxNumOfTransformations,
      String workingJavaPackage,
      List<JarTransformationRecord> jarTransformationRecords,
      ImmutableList<Path> bootClassPathEntries,
      ImmutableListMultimap<String, String> customCommandOptions,
      ImmutableList<Path> inputJars,
      ImmutableList<Path> classPathEntries,
      ImmutableList<Field> injectableClassLiterals,
      ImmutableList<Field> injectableAsmNodes,
      ImmutableList<Field> injectableMethodHandles,
      Path androidRuntimeJar,
      Path jacocoAgentJar) {
    this.testInstance = testInstance;
    this.testInstanceLookup = testInstanceLookup;

    this.maxNumOfTransformations = maxNumOfTransformations;
    this.workingJavaPackage = workingJavaPackage;
    this.jarTransformationRecords = jarTransformationRecords;

    this.inputs = inputJars;
    this.classPathEntries = classPathEntries;
    this.bootClassPathEntries = bootClassPathEntries;
    this.extraCustomCommandOptions = customCommandOptions;

    this.injectableClassLiterals = injectableClassLiterals;
    this.injectableAsmNodes = injectableAsmNodes;
    this.injectableMethodHandles = injectableMethodHandles;
    this.injectableZipEntries = injectableZipEntries;
    this.androidRuntimeJar = androidRuntimeJar;
    this.jacocoAgentJar = jacocoAgentJar;

    checkState(Files.exists(androidRuntimeJar));
    checkState(Files.exists(jacocoAgentJar));
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

            ClassLoader inputClassLoader = getInputClassLoader();
            for (Field field : injectableClassLiterals) {
              Class<?> classLiteral =
                  loadClassLiteral(
                      field.getDeclaredAnnotation(DynamicClassLiteral.class),
                      jarTransformationRecords,
                      inputClassLoader,
                      reflectionBasedMembers,
                      descriptorLookupRepo,
                      workingJavaPackage);
              MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
              fieldSetter.invoke(testInstance, classLiteral);
            }

            for (Field field : injectableAsmNodes) {
              Class<?> requestedFieldType = field.getType();
              Object asmNode =
                  getAsmNode(
                      field.getDeclaredAnnotation(AsmNode.class),
                      requestedFieldType,
                      jarTransformationRecords,
                      inputs,
                      workingJavaPackage);
              MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
              fieldSetter.invoke(testInstance, asmNode);
            }

            for (Field field : injectableMethodHandles) {
              MethodHandle methodHandle =
                  getMethodHandle(
                      field.getDeclaredAnnotation(RuntimeMethodHandle.class),
                      testInstanceLookup,
                      jarTransformationRecords,
                      inputClassLoader,
                      reflectionBasedMembers,
                      descriptorLookupRepo,
                      workingJavaPackage);
              MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
              fieldSetter.invoke(testInstance, methodHandle);
            }

            for (Field field : injectableZipEntries) {
              ZipEntry zipEntry =
                  getZipEntry(
                      field.getDeclaredAnnotation(RuntimeZipEntry.class),
                      jarTransformationRecords,
                      inputs,
                      workingJavaPackage);
              MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
              fieldSetter.invoke(testInstance, zipEntry);
            }
            base.evaluate();
          }
        },
        description);
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

  @AutoAnnotation
  private static DynamicClassLiteral createLoadClassLiteralRequest(String value, int round) {
    return new AutoAnnotation_DesugarRule_createLoadClassLiteralRequest(value, round);
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
            createLoadClassLiteralRequest(methodHandleRequest.className(), round),
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
        outputDirPath = Paths.get(temporaryFolder.newFolder(targetDirKey.split("/")).getPath());
        tempDirs.put(targetDirKey, outputDirPath);
      }
      outputRuntimePathsBuilder.add(outputDirPath.resolve(path.getFileName()));
    }
    return outputRuntimePathsBuilder.build();
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

}
