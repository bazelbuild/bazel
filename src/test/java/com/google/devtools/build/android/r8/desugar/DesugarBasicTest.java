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
package com.google.devtools.build.android.r8.desugar;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.android.r8.R8Utils.DESUGAR_INTERFACE_COMPANION_SUFFIX;
import static com.google.devtools.build.android.r8.R8Utils.INTERFACE_COMPANION_SUFFIX;
import static org.objectweb.asm.Opcodes.V1_7;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.proto.DesugarDeps;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.Dependency;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.DesugarDepsInfo;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.InterfaceDetails;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.InterfaceWithCompanion;
import com.google.devtools.build.android.r8.DescriptorUtils;
import com.google.devtools.build.android.r8.Desugar;
import com.google.devtools.build.android.r8.FileUtils;
import com.google.devtools.build.android.r8.desugar.basic.A;
import com.google.devtools.build.android.r8.desugar.basic.B;
import com.google.devtools.build.android.r8.desugar.basic.C;
import com.google.devtools.build.android.r8.desugar.basic.I;
import com.google.devtools.build.android.r8.desugar.basic.J;
import com.google.devtools.build.android.r8.desugar.basic.K;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.function.Consumer;
import java.util.jar.JarEntry;
import java.util.jar.JarInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;

/** Basic test of D8 desugar */
@RunWith(JUnit4.class)
public class DesugarBasicTest {
  private Path basic;
  private Path desugared;
  private Path desugaredClasspath;
  private Path desugaredWithDependencyMetadata;
  private Path desugaredWithDependencyMetadataWithDesugar;
  private Path doubleDesugaredWithDependencyMetadata;
  private Path desugaredClasspathWithDependencyMetadata;
  private Path desugaredClasspathWithDependencyMetadataWithDesugar;
  private Path desugaredWithDependencyMetadataMissingInterface;
  private Path desugaredWithDependencyMetadataMissingInterfaceWithDesugar;

  @Before
  public void setup() {
    // Jar file with the compiled Java code in the sub-package basic before desugaring.
    basic = Paths.get(System.getProperty("DesugarBasicTest.testdata_basic"));
    // Jar file with the compiled Java code in the sub-package basic after desugaring.
    desugared = Paths.get(System.getProperty("DesugarBasicTest.testdata_basic_desugared"));
    // Jar file with the compiled Java code in the sub-package basic after desugaring with all
    // interfaces on classpath.
    desugaredClasspath =
        Paths.get(System.getProperty("DesugarBasicTest.testdata_basic_desugared_classpath"));
    // Jar file with the compiled Java code in the sub-package basic after desugaring with
    // collected dependency metadata included.
    desugaredWithDependencyMetadata =
        Paths.get(
            System.getProperty(
                "DesugarBasicTest.testdata_basic_desugared_with_dependency_metadata"));
    // Same as above compiled with desugar.
    desugaredWithDependencyMetadataWithDesugar =
        Paths.get(
            System.getProperty(
                "DesugarBasicTest.testdata_basic_desugared_with_dependency_metadata_with_desugar"));
    // Same as testdata_basic_desugared_with_dependency_metadata, but where the input is instead the
    // already desugared code (i.e., testdata_basic_desugared)
    doubleDesugaredWithDependencyMetadata =
        Paths.get(
            System.getProperty(
                "DesugarBasicTest.testdata_basic_double_desugared_with_dependency_metadata"));
    // Jar file with the compiled Java code in the sub-package basic after desugaring with all
    // interfaces on classpath with collected dependency metadata included.
    desugaredClasspathWithDependencyMetadata =
        Paths.get(
            System.getProperty(
                "DesugarBasicTest.testdata_basic_desugared_classpath_with_dependency_metadata"));
    // Same as above compiled with desugar.
    desugaredClasspathWithDependencyMetadataWithDesugar =
        Paths.get(
            System.getProperty(
                "DesugarBasicTest.testdata_basic_desugared_classpath_with_dependency_metadata_with_desugar"));
    // Jar file with the compiled Java code in the sub-package basic after desugaring with missing
    // interface.
    desugaredWithDependencyMetadataMissingInterface =
        Paths.get(
            System.getProperty(
                "DesugarBasicTest.testdata_basic_desugared_with_dependency_metadata_missing_interface"));
    // Same as above compiled with desugar.
    desugaredWithDependencyMetadataMissingInterfaceWithDesugar =
        Paths.get(
            System.getProperty(
                "DesugarBasicTest.testdata_basic_desugared_with_dependency_metadata_missing_interface_with_desugar"));
  }

  @Test
  public void checkBeforeDesugar() throws Exception {
    DesugarInfoCollector desugarInfoCollector = new DesugarInfoCollector();
    forAllClasses(basic, desugarInfoCollector);
    assertThat(desugarInfoCollector.getLargestMajorClassFileVersion()).isGreaterThan(V1_7);
    assertThat(desugarInfoCollector.getNumberOfInvokeDynamic()).isGreaterThan(0);
    assertThat(desugarInfoCollector.getNumberOfDefaultMethods()).isGreaterThan(0);
    assertThat(desugarInfoCollector.getNumberOfDesugaredLambdas()).isEqualTo(0);
    assertThat(desugarInfoCollector.getNumberOfCompanionClasses()).isEqualTo(0);
  }

  @Test
  public void checkAfterDesugar() throws Exception {
    for (Path jar : ImmutableList.of(desugared, desugaredWithDependencyMetadata)) {
      DesugarInfoCollector desugarInfoCollector = new DesugarInfoCollector();
      forAllClasses(jar, desugarInfoCollector);
      // TODO(b/153971249): The class file version of desugared class files should be Java 7.
      // assertThat(lambdaUse.getMajorCfVersion()).isEqualTo(V1_7);
      assertThat(desugarInfoCollector.getNumberOfInvokeDynamic()).isEqualTo(0);
      assertThat(desugarInfoCollector.getNumberOfDefaultMethods()).isEqualTo(0);
      assertThat(desugarInfoCollector.getNumberOfDesugaredLambdas()).isEqualTo(1);
      assertThat(desugarInfoCollector.getNumberOfCompanionClasses()).isEqualTo(3);
    }
  }

  @Test
  public void checkAfterDesugarClasspath() throws Exception {
    DesugarInfoCollector desugarInfoCollector = new DesugarInfoCollector();
    forAllClasses(desugaredClasspath, desugarInfoCollector);
    // TODO(b/153971249): The class file version of desugared class files should be Java 7.
    // assertThat(lambdaUse.getMajorCfVersion()).isEqualTo(V1_7);
    assertThat(desugarInfoCollector.getNumberOfInvokeDynamic()).isEqualTo(0);
    assertThat(desugarInfoCollector.getNumberOfDefaultMethods()).isEqualTo(0);
    assertThat(desugarInfoCollector.getNumberOfDesugaredLambdas()).isEqualTo(1);
    assertThat(desugarInfoCollector.getNumberOfCompanionClasses()).isEqualTo(0);
  }

  @Test
  public void checkMetaDataAfterDoubleDesugaring() throws Exception {
    DesugarDepsInfo info = extractDesugarDeps(doubleDesugaredWithDependencyMetadata);
    assertThat(info.getInterfaceWithCompanionCount()).isEqualTo(0);
    assertThat(info.getAssumePresentCount()).isEqualTo(0);
    assertThat(info.getMissingInterfaceCount()).isEqualTo(0);
    assertThat(info.getInterfaceWithSupertypesList())
        .containsExactly(
            InterfaceDetails.newBuilder()
                .setOrigin(classToType(J.class))
                .addExtendedInterface(classToType(I.class))
                .build());
  }

  @SuppressWarnings("ProtoParseWithRegistry")
  private static DesugarDepsInfo extractDesugarDeps(Path jar) throws Exception {
    try (ZipFile zip = new ZipFile(jar.toFile())) {
      ZipEntry desugarDepsEntry = zip.getEntry(Desugar.DESUGAR_DEPS_FILENAME);
      assertThat(desugarDepsEntry).isNotNull();
      return DesugarDepsInfo.parseFrom(zip.getInputStream(desugarDepsEntry));
    }
  }

  private static DesugarDepsInfo mapD8Info(DesugarDepsInfo info) {
    // Rebuild the D8 dependency information with the same naming scheme for companion classes
    // as desugar use.
    DesugarDepsInfo.Builder builder = info.newBuilderForType();
    info.getAssumePresentList()
        .forEach(
            assumePresent ->
                builder.addAssumePresent(
                    assumePresent
                        .newBuilderForType()
                        .setOrigin(assumePresent.getOrigin())
                        .setTarget(
                            assumePresent
                                .getTarget()
                                .newBuilderForType()
                                .setBinaryName(
                                    assumePresent
                                        .getTarget()
                                        .getBinaryName()
                                        .replace(
                                            INTERFACE_COMPANION_SUFFIX,
                                            DESUGAR_INTERFACE_COMPANION_SUFFIX)))));
    info.getInterfaceWithCompanionList().forEach(builder::addInterfaceWithCompanion);
    info.getInterfaceWithSupertypesList().forEach(builder::addInterfaceWithSupertypes);
    info.getMissingInterfaceList().forEach(builder::addMissingInterface);
    return builder.build();
  }

  private static DesugarDeps.Type classToType(Class<?> clazz) {
    return DesugarDeps.Type.newBuilder()
        .setBinaryName(DescriptorUtils.classToBinaryName(clazz))
        .build();
  }

  private static DesugarDeps.Type classToCompanionType(Class<?> clazz) {
    return DesugarDeps.Type.newBuilder()
        .setBinaryName(DescriptorUtils.classToBinaryName(clazz) + INTERFACE_COMPANION_SUFFIX)
        .build();
  }

  @Test
  public void checkDependencyMetadata() throws Exception {
    DesugarDepsInfo info = extractDesugarDeps(desugaredWithDependencyMetadata);

    // Check expected metadata content.
    assertThat(info.getAssumePresentList())
        .containsExactly(
            Dependency.newBuilder()
                .setOrigin(classToType(A.class))
                .setTarget(classToCompanionType(I.class))
                .build(),
            Dependency.newBuilder()
                .setOrigin(classToType(B.class))
                .setTarget(classToCompanionType(J.class))
                .build(),
            Dependency.newBuilder()
                .setOrigin(classToType(C.class))
                .setTarget(classToCompanionType(K.class))
                .build());
    assertThat(info.getInterfaceWithCompanionList())
        .containsExactly(
            InterfaceWithCompanion.newBuilder()
                .setOrigin(classToType(I.class))
                .setNumDefaultMethods(1)
                .build(),
            InterfaceWithCompanion.newBuilder()
                .setOrigin(classToType(J.class))
                .setNumDefaultMethods(1)
                .build(),
            InterfaceWithCompanion.newBuilder()
                .setOrigin(classToType(K.class))
                .setNumDefaultMethods(1)
                .build());
    assertThat(info.getInterfaceWithSupertypesList())
        .containsExactly(
            InterfaceDetails.newBuilder()
                .setOrigin(classToType(J.class))
                .addExtendedInterface(classToType(I.class))
                .build());
    assertThat(info.getMissingInterfaceCount()).isEqualTo(0);

    // Compare metadata with desugar metadata.
    assertThat(mapD8Info(info))
        .isEqualTo(extractDesugarDeps(desugaredWithDependencyMetadataWithDesugar));
  }

  @Test
  public void checkDependencyMetadataClasspath() throws Exception {
    DesugarDepsInfo info = extractDesugarDeps(desugaredClasspathWithDependencyMetadata);

    // Check expected metadata content.
    assertThat(info.getAssumePresentList())
        .containsExactly(
            Dependency.newBuilder()
                .setOrigin(classToType(A.class))
                .setTarget(classToCompanionType(I.class))
                .build(),
            Dependency.newBuilder()
                .setOrigin(classToType(B.class))
                .setTarget(classToCompanionType(J.class))
                .build(),
            Dependency.newBuilder()
                .setOrigin(classToType(C.class))
                .setTarget(classToCompanionType(K.class))
                .build());
    assertThat(info.getInterfaceWithCompanionCount()).isEqualTo(0);
    assertThat(info.getInterfaceWithSupertypesCount()).isEqualTo(0);
    assertThat(info.getMissingInterfaceCount()).isEqualTo(0);

    // Compare metadata with desugar metadata.
    assertThat(mapD8Info(info))
        .isEqualTo(extractDesugarDeps(desugaredClasspathWithDependencyMetadataWithDesugar));
  }

  @Test
  public void checkDependencyMetadataMissingInterface() throws Exception {
    DesugarDepsInfo info = extractDesugarDeps(desugaredWithDependencyMetadataMissingInterface);

    // Check expected metadata content.
    assertThat(info.getAssumePresentList())
        .containsExactly(
            Dependency.newBuilder()
                .setOrigin(classToType(A.class))
                .setTarget(classToCompanionType(I.class))
                .build(),
            Dependency.newBuilder()
                .setOrigin(classToType(B.class))
                .setTarget(classToCompanionType(J.class))
                .build());
    assertThat(info.getInterfaceWithCompanionList())
        .containsExactly(
            InterfaceWithCompanion.newBuilder()
                .setOrigin(classToType(I.class))
                .setNumDefaultMethods(1)
                .build(),
            InterfaceWithCompanion.newBuilder()
                .setOrigin(classToType(J.class))
                .setNumDefaultMethods(1)
                .build());
    assertThat(info.getInterfaceWithSupertypesList())
        .containsExactly(
            InterfaceDetails.newBuilder()
                .setOrigin(classToType(J.class))
                .addExtendedInterface(classToType(I.class))
                .build());
    assertThat(info.getMissingInterfaceList())
        .containsExactly(
            Dependency.newBuilder()
                .setOrigin(classToType(C.class))
                .setTarget(classToType(K.class))
                .build());

    // Compare metadata with desugar metadata.
    assertThat(mapD8Info(info))
        .isEqualTo(extractDesugarDeps(desugaredWithDependencyMetadataMissingInterfaceWithDesugar));
  }

  private static void forAllClasses(Path jar, ClassVisitor classVisitor) throws Exception {
    forAllClasses(
        jar,
        classReader ->
            classReader.accept(classVisitor, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES));
  }

  private static void forAllClasses(Path jar, Consumer<ClassReader> classReader) throws Exception {

    try (JarInputStream jarInputStream =
        new JarInputStream(Files.newInputStream(jar, StandardOpenOption.READ))) {
      JarEntry entry;
      while ((entry = jarInputStream.getNextJarEntry()) != null) {
        String entryName = entry.getName();
        if (FileUtils.isClassFile(entryName)) {
          classReader.accept(new ClassReader(jarInputStream));
        }
      }
    }
  }
}
