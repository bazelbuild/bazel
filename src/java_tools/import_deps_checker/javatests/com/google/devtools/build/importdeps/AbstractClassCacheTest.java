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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/** Base class for {@link ClassCacheTest} and {@link DepsCheckerClassVisitorTest}. */
public abstract class AbstractClassCacheTest {

  static final String PACKAGE_NAME = "com/google/devtools/build/importdeps/testdata/";

  final Path bootclasspath = getPathFromSystemProperty("classcache.test.bootclasspath");

  final Path clientJar = getPathFromSystemProperty("classcache.test.Client");
  final ImmutableList<String> clientJarPositives =
      ImmutableList.of("Client", "Client$NestedAnnotation")
          .stream()
          .map(s -> PACKAGE_NAME + s)
          .collect(ImmutableList.toImmutableList());

  final Path libraryJar = getPathFromSystemProperty("classcache.test.Library");
  final ImmutableList<String> libraryJarPositives =
      ImmutableList.<String>builder().add("Library")
          .addAll(
              IntStream.range(1, 13)
                  .mapToObj(i -> "Library$Class" + i)
                  .collect(ImmutableList.toImmutableList()))
          .build().stream()
          .map(s -> PACKAGE_NAME + s)
          .collect(ImmutableList.toImmutableList());

  final Path libraryWoMembersJar = getPathFromSystemProperty("classcache.test.Library_no_members");

  final Path libraryAnnotationsJar =
      getPathFromSystemProperty("classcache.test.LibraryAnnotations");
  final ImmutableList<String> libraryAnnotationsJarPositives =
      ImmutableList.<String>builder()
          .add("LibraryAnnotations")
          .addAll(
              Stream.of(
                      "ClassAnnotation",
                      "MethodAnnotation",
                      "FieldAnnotation",
                      "ConstructorAnnotation",
                      "ParameterAnnotation",
                      "TypeAnnotation",
                      "AnnotationAnnotation",
                      "AnnotationFlag")
                  .map(name -> "LibraryAnnotations$" + name)
                  .collect(ImmutableList.toImmutableList()))
          .build()
          .stream()
          .map(s -> PACKAGE_NAME + s)
          .collect(ImmutableList.toImmutableList());

  final Path libraryExceptionJar = getPathFromSystemProperty("classcache.test.LibraryException");
  final ImmutableList<String> libraryExceptionJarPositives =
      ImmutableList.of(PACKAGE_NAME + "LibraryException");

  final Path libraryInterfaceJar = getPathFromSystemProperty("classcache.test.LibraryInterface");
  final ImmutableList<String> libraryInterfacePositives =
      ImmutableList.of(
          PACKAGE_NAME + "LibraryInterface",
          PACKAGE_NAME + "LibraryInterface$Func",
          PACKAGE_NAME + "LibraryInterface$One",
          PACKAGE_NAME + "LibraryInterface$Two",
          PACKAGE_NAME + "LibraryInterface$InterfaceFoo",
          PACKAGE_NAME + "LibraryInterface$InterfaceBar");

  final Path libraryModuleInfoJar = getPathFromSystemProperty("classcache.test.LibraryModuleInfo");

  static Path getPathFromSystemProperty(String propertyName) {
    String path =
        checkNotNull(
            System.getProperty(propertyName), "The system property %s is not set.", propertyName);
    return Paths.get(path);
  }

  /** Flattern a list of lists. */
  static <T> ImmutableList<T> combine(ImmutableList<T>... lists) {
    return Arrays.stream(lists)
        .flatMap(ImmutableList::stream)
        .collect(ImmutableList.toImmutableList());
  }
}
