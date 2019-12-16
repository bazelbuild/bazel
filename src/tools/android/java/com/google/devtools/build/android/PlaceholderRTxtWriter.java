// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.android.resources.ResourceType;
import java.io.BufferedWriter;
import java.io.Flushable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Generates an R.txt file (with dummy values) declaring the fields used by a library.
 *
 * <p>The real IDs will be collected by the R.txt file from aapt2 and joined via {@code
 * com.google.devtools.build.android.resources.FieldInitializers}.
 */
final class PlaceholderRTxtWriter implements Flushable, AndroidResourceSymbolSink {

  private final Path rTxtOut;

  private final Map<ResourceType, Set<String>> innerClasses = new EnumMap<>(ResourceType.class);
  private final Map<String, Set</*attr=*/ String>> styleables = new TreeMap<>();

  static PlaceholderRTxtWriter create(Path rTxtOut) {
    return new PlaceholderRTxtWriter(rTxtOut);
  }

  private PlaceholderRTxtWriter(Path rTxtOut) {
    this.rTxtOut = rTxtOut;
  }

  @Override
  public void acceptSimpleResource(DependencyInfo unused, ResourceType type, String name) {
    innerClasses
        .computeIfAbsent(type, t -> new TreeSet<>())
        .add(PlaceholderIdFieldInitializerBuilder.normalizeName(name));
  }

  @Override
  public void acceptStyleableResource(
      DependencyInfo dependencyInfo,
      FullyQualifiedName key,
      Map<FullyQualifiedName, Boolean> attrs) {
    Set<String> attrSet =
        styleables.computeIfAbsent(
            PlaceholderIdFieldInitializerBuilder.normalizeName(key.name()), n -> new TreeSet<>());

    for (FullyQualifiedName attr : attrs.keySet()) {
      attrSet.add(PlaceholderIdFieldInitializerBuilder.normalizeAttrName(attr));
    }
  }

  @Override
  public void flush() throws IOException {
    try (BufferedWriter writer = Files.newBufferedWriter(rTxtOut, UTF_8)) {
      for (Map.Entry<ResourceType, Set<String>> innerClass : innerClasses.entrySet()) {
        ResourceType resourceType = innerClass.getKey();
        for (String fieldName : innerClass.getValue()) {
          writer.write(String.format("int %s %s 0\n", resourceType.getName(), fieldName));
        }
      }

      for (Map.Entry<String, Set<String>> styleable : styleables.entrySet()) {
        // NB: the size of this array is irrelevant, we just need to declare it as one.
        writer.write(String.format("int[] styleable %s { 0 }\n", styleable.getKey()));

        for (String attr : styleable.getValue()) {
          writer.write(String.format("int styleable %s_%s 0\n", styleable.getKey(), attr));
        }
      }
    }
  }
}
