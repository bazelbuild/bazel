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

package com.google.devtools.build.android.desugar.langmodel;

import com.google.common.collect.ImmutableSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/** Tracks {@link ClassAttributes} for all classes under investigation. */
public class ClassAttributeRecord {

  private final Map<String, ClassAttributes> record = new HashMap<>();

  public static ClassAttributeRecord create() {
    return new ClassAttributeRecord();
  }

  private ClassAttributeRecord() {}

  public ClassAttributes setClassAttributes(ClassAttributes classAttributes) {
    String classBinaryName = classAttributes.classBinaryName();
    if (record.containsKey(classBinaryName)) {
      throw new IllegalStateException(
          String.format(
              "Expected the ClassAttributes of a class to be put into this record only once during"
                  + " {@link ClassVisitor#visitEnd}: Pre-existing: (%s), Now (%s)",
              record.get(classBinaryName), classAttributes));
    }
    return record.put(classBinaryName, classAttributes);
  }

  public Optional<String> getNestHost(String className) {
    ClassAttributes classAttributes = record.get(className);
    if (classAttributes != null) {
      return classAttributes.nestHost();
    }
    return Optional.empty();
  }

  public ImmutableSet<String> getNestMembers(String className) {
    ClassAttributes classAttributes = record.get(className);
    if (classAttributes != null) {
      return classAttributes.nestMembers();
    }
    return ImmutableSet.of();
  }
}
