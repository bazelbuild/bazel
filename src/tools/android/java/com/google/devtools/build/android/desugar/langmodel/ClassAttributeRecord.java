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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/** Tracks {@link ClassAttributes} for all classes under investigation. */
public final class ClassAttributeRecord implements TypeMappable<ClassAttributeRecord> {

  private final Map<ClassName, ClassAttributes> record;

  public static ClassAttributeRecord create() {
    return new ClassAttributeRecord(new HashMap<>());
  }

  private ClassAttributeRecord(Map<ClassName, ClassAttributes> record) {
    this.record = record;
  }

  public ClassAttributes setClassAttributes(ClassAttributes classAttributes) {
    ClassName classBinaryName = classAttributes.classBinaryName();
    if (record.containsKey(classBinaryName)) {
      throw new IllegalStateException(
          String.format(
              "Expected the ClassAttributes of a class to be put into this record only once during"
                  + " {@link ClassVisitor#visitEnd}: Pre-existing: (%s), Now (%s)",
              record.get(classBinaryName), classAttributes));
    }
    return record.put(classBinaryName, classAttributes);
  }

  public Optional<ClassName> getNestHost(ClassName className) {
    ClassAttributes classAttributes = record.get(className);
    checkNotNull(
        classAttributes,
        "Expected recorded ClassAttributes for (%s). Available record: %s",
        className,
        record.keySet());
    return classAttributes.nestHost();
  }

  public ImmutableSet<ClassName> getNestMembers(ClassName className) {
    ClassAttributes classAttributes = record.get(className);
    checkNotNull(
        classAttributes,
        "Expected recorded ClassAttributes for (%s). Available record: %s",
        className,
        record.keySet());
    return classAttributes.nestMembers();
  }

  @Override
  public ClassAttributeRecord acceptTypeMapper(TypeMapper typeMapper) {
    return new ClassAttributeRecord(typeMapper.mapMutable(record));
  }
}
