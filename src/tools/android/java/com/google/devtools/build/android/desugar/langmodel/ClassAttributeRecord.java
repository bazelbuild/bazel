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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.Optional;

/** Tracks {@link ClassAttributes} for all classes under investigation. */
@AutoValue
public abstract class ClassAttributeRecord implements TypeMappable<ClassAttributeRecord> {

  public abstract ImmutableMap<ClassName, ClassAttributes> record();

  public static ClassAttributeRecordBuilder builder() {
    return new AutoValue_ClassAttributeRecord.Builder();
  }

  public final Optional<ClassName> getNestHost(ClassName className) {
    ClassAttributes classAttributes = record().get(className);
    checkNotNull(
        classAttributes,
        "Expected recorded ClassAttributes for (%s). Available record: %s",
        className,
        record().keySet());
    return classAttributes.nestHost();
  }

  public final ImmutableSet<ClassName> getNestMembers(ClassName className) {
    ClassAttributes classAttributes = record().get(className);
    checkNotNull(
        classAttributes,
        "Expected recorded ClassAttributes for (%s). Available record: %s",
        className,
        record().keySet());
    return classAttributes.nestMembers();
  }

  @Override
  public final ClassAttributeRecord acceptTypeMapper(TypeMapper typeMapper) {
    return ClassAttributeRecord.builder().setRecord(typeMapper.map(record())).build();
  }

  /** The builder for {@link ClassAttributeRecord}. */
  @AutoValue.Builder
  public abstract static class ClassAttributeRecordBuilder {

    abstract ImmutableMap.Builder<ClassName, ClassAttributes> recordBuilder();

    abstract ClassAttributeRecordBuilder setRecord(ImmutableMap<ClassName, ClassAttributes> record);

    public final ClassAttributeRecordBuilder addClassAttributes(ClassAttributes classAttributes) {
      recordBuilder().put(classAttributes.classBinaryName(), classAttributes);
      return this;
    }

    public abstract ClassAttributeRecord build();
  }
}
