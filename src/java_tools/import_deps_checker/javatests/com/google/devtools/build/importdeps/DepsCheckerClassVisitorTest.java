// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.importdeps;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.importdeps.ResultCollector.MissingMember;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;

/** Test for {@link DepsCheckerClassVisitor}. */
@RunWith(JUnit4.class)
public class DepsCheckerClassVisitorTest extends AbstractClassCacheTest {

  @Test
  public void testMissingLibraryException() throws IOException {
    assertThat(
            getMissingClassesInClient(
                bootclasspath, libraryJar, libraryInterfaceJar, libraryAnnotationsJar, clientJar))
        .containsExactlyElementsIn(libraryExceptionJarPositives);
  }

  @Test
  public void testMissingLibraryInterface() throws IOException {
    assertThat(
            getMissingClassesInClient(
                bootclasspath, libraryJar, libraryAnnotationsJar, libraryExceptionJar, clientJar))
        .containsExactlyElementsIn(libraryInterfacePositives);
  }

  @Test
  public void testMissingLibraryAnnotations() throws IOException {
    assertThat(
            getMissingClassesInClient(
                bootclasspath, libraryJar, libraryExceptionJar, libraryInterfaceJar, clientJar))
        .containsExactlyElementsIn(libraryAnnotationsJarPositives);
  }

  @Test
  public void testMissingLibraryInClient() throws IOException {
    assertThat(
            getMissingClassesInClient(
                bootclasspath,
                libraryExceptionJar,
                libraryInterfaceJar,
                libraryAnnotationsJar,
                clientJar))
        .containsExactlyElementsIn(libraryJarPositives);
  }

  @Test
  public void testMissingMembersInClient() throws IOException {
    ResultCollector collector =
        getResultCollector(
            /*checkMissingMembers=*/ true,
            bootclasspath,
            libraryAnnotationsJar,
            libraryInterfaceJar,
            libraryWoMembersJar,
            libraryExceptionJar,
            clientJar);
    assertThat(collector.getSortedMissingClassInternalNames()).isEmpty();
    assertThat(
            collector
                .getSortedMissingMembers()
                .stream()
                .map(DepsCheckerClassVisitorTest::constructFullQualifiedMemberName)
                .collect(ImmutableList.toImmutableList()))
        .containsExactly(
            constructFullyQualifiedMemberName(
                "com/google/devtools/build/importdeps/testdata/Library$Class1",
                "I",
                "Lcom/google/devtools/build/importdeps/testdata/Library$Class1;"),
            constructFullyQualifiedMemberName(
                "com/google/devtools/build/importdeps/testdata/Library$Class3",
                "field",
                "Lcom/google/devtools/build/importdeps/testdata/Library$Class4;"),
            constructFullyQualifiedMemberName(
                "com/google/devtools/build/importdeps/testdata/Library$Class4",
                "createClass5",
                "()Lcom/google/devtools/build/importdeps/testdata/Library$Class5;"),
            constructFullyQualifiedMemberName(
                "com/google/devtools/build/importdeps/testdata/Library$Class5",
                "create",
                "(Lcom/google/devtools/build/importdeps/testdata/Library$Class7;)"
                    + "Lcom/google/devtools/build/importdeps/testdata/Library$Class6;"))
        .inOrder();
  }

  @Test
  public void testMissingMembersIgnoredWhenUnchecked() throws IOException {
    ResultCollector collector =
        getResultCollector(
            /*checkMissingMembers=*/ false,
            bootclasspath,
            libraryAnnotationsJar,
            libraryInterfaceJar,
            libraryWoMembersJar,
            libraryExceptionJar,
            clientJar);
    assertThat(collector.isEmpty()).isTrue();
  }

  @Test
  public void testSafelyHandleModuleInfo() throws IOException {
    // We don't need to assert anything-- we just need to make sure the tool
    // can safely handle odd classes like module-info without throwing any
    // exceptions.
    // module-info's peculiarity is that it's like java.lang.Object
    // and doesn't have a superclass.
    try (ZipFile zipFile = new ZipFile(libraryModuleInfoJar.toFile())) {
      ZipEntry entry = zipFile.getEntry("module-info.class");
      ClassCache cache =
          new ClassCache(
              ImmutableSet.of(), ImmutableSet.of(), ImmutableSet.of(), ImmutableSet.of(), false);
      try (InputStream classStream = zipFile.getInputStream(entry)) {
        ClassReader reader = new ClassReader(classStream);
        DepsCheckerClassVisitor checker = new DepsCheckerClassVisitor(cache, null);
        reader.accept(checker, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
      }
    }
  }

  private static String constructFullQualifiedMemberName(MissingMember member) {
    return constructFullyQualifiedMemberName(
        member.owner().internalName(), member.memberName(), member.descriptor());
  }

  private static String constructFullyQualifiedMemberName(
      String owner, String memberName, String descriptor) {
    return owner + memberName + descriptor;
  }

  private ImmutableList<String> getMissingClassesInClient(Path... classpath) throws IOException {
    ResultCollector resultCollector = getResultCollector(/*checkMissingMembers=*/ false, classpath);
    return resultCollector.getSortedMissingClassInternalNames();
  }

  private ResultCollector getResultCollector(boolean checkMissingMembers, Path... classpath)
      throws IOException {
    ImmutableList<String> clientClasses =
        ImmutableList.of(
            PACKAGE_NAME + "Client",
            PACKAGE_NAME + "Client$NestedAnnotation",
            PACKAGE_NAME + "Client$InnerClassWithSyntheticConstructorParam");
    ResultCollector resultCollector = new ResultCollector(checkMissingMembers);
    try (ClassCache cache =
            new ClassCache(
                ImmutableSet.copyOf(classpath),
                ImmutableSet.of(),
                ImmutableSet.of(),
                ImmutableSet.of(),
                checkMissingMembers);
        ZipFile zipFile = new ZipFile(clientJar.toFile())) {
      assertThat(cache.getClassState("java/lang/invoke/LambdaMetafactory").isExistingState())
          .isTrue();
      AbstractClassEntryState state = cache.getClassState("java/lang/Enum");
      assertThat(state.isExistingState()).isTrue();
      for (String clientClass : clientClasses) {
        ZipEntry entry = zipFile.getEntry(clientClass + ".class");
        try (InputStream classStream = zipFile.getInputStream(entry)) {
          ClassReader reader = new ClassReader(classStream);
          DepsCheckerClassVisitor checker = new DepsCheckerClassVisitor(cache, resultCollector);
          reader.accept(checker, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
        }
      }
    }
    return resultCollector;
  }
}
