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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import java.util.Optional;

/**
 * A class to hold class-level attribute information.
 *
 * <p>https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-4.html#jvms-4.7
 */
@AutoValue
public abstract class ClassAttributes implements TypeMappable<ClassAttributes> {

  public abstract ClassName classBinaryName();

  public abstract int majorVersion();

  public abstract Optional<ClassName> nestHost();

  public abstract ImmutableSet<ClassName> nestMembers();

  public abstract ImmutableSet<MethodKey> privateInstanceMethods();

  public abstract ImmutableSet<MethodKey> desugarIgnoredMethods();

  // Include other class attributes as necessary.

  public static ClassAttributesBuilder builder() {
    return new AutoValue_ClassAttributes.Builder();
  }

  @Override
  public ClassAttributes acceptTypeMapper(TypeMapper typeMapper) {
    ClassAttributesBuilder mappedBuilder = builder();
    mappedBuilder.setClassBinaryName(classBinaryName().acceptTypeMapper(typeMapper));
    mappedBuilder.setMajorVersion(majorVersion());
    if (nestHost().isPresent()) {
      mappedBuilder.setNestHost(nestHost().get().acceptTypeMapper(typeMapper));
    }
    nestMembers().stream().map(typeMapper::map).forEach(mappedBuilder::addNestMember);
    privateInstanceMethods().stream()
        .map(methodKey -> methodKey.acceptTypeMapper(typeMapper))
        .forEach(mappedBuilder::addPrivateInstanceMethod);
    desugarIgnoredMethods().stream()
        .map(methodKey -> methodKey.acceptTypeMapper(typeMapper))
        .forEach(mappedBuilder::addDesugarIgnoredMethods);
    mappedBuilder.setClassBinaryName(classBinaryName().acceptTypeMapper(typeMapper));
    return mappedBuilder.build();
  }

  /** The builder of {@link ClassAttributes}. */
  @AutoValue.Builder
  public abstract static class ClassAttributesBuilder {

    public abstract ClassAttributesBuilder setClassBinaryName(ClassName classBinaryName);

    public abstract ClassAttributesBuilder setMajorVersion(int value);

    public abstract ClassAttributesBuilder setNestHost(ClassName nestHost);

    abstract ImmutableSet.Builder<ClassName> nestMembersBuilder();

    abstract ImmutableSet.Builder<MethodKey> privateInstanceMethodsBuilder();

    abstract ImmutableSet.Builder<MethodKey> desugarIgnoredMethodsBuilder();

    public ClassAttributesBuilder addNestMember(ClassName nestMember) {
      nestMembersBuilder().add(nestMember);
      return this;
    }

    public ClassAttributesBuilder addPrivateInstanceMethod(MethodKey methodKey) {
      privateInstanceMethodsBuilder().add(methodKey);
      return this;
    }

    public ClassAttributesBuilder addDesugarIgnoredMethods(MethodKey methodKey) {
      desugarIgnoredMethodsBuilder().add(methodKey);
      return this;
    }

    public abstract ClassAttributes build();
  }
}
