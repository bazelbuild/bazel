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

package com.google.devtools.build.android.desugar.langmodel;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberTrackReason.ClassMemberTrackReasonBuilder;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Predicate;

/**
 * A record that tracks the declarations and usages of a class member, including fields,
 * constructors and methods.
 */
@AutoValue
public abstract class ClassMemberRecord implements TypeMappable<ClassMemberRecord> {

  /** Tracks a class member with a reason. */
  abstract ImmutableMap<ClassMemberKey<?>, ClassMemberTrackReason> reasons();

  /** Creates a builder instance for this class. */
  public static ClassMemberRecordBuilder builder() {
    return new AutoValue_ClassMemberRecord.Builder();
  }

  /** Gets class members with both tracked declarations and tracked usage. */
  public final ClassMemberRecord filterUsedMemberWithTrackedDeclaration() {
    return builder()
        .setReasons(
            reasons().entrySet().stream()
                .filter(
                    entry -> {
                      ClassMemberTrackReason reason = entry.getValue();
                      // keep interface members and class members with use references.
                      return reason.hasInterfaceDeclReason()
                          || (reason.hasDeclReason() && reason.hasMemberUseReason());
                    })
                .collect(toImmutableMap(Map.Entry::getKey, Map.Entry::getValue)))
        .autoInternalBuild();
  }

  /** Find all member keys that represent a constructor. */
  public final ImmutableList<ClassMemberKey<?>> findAllConstructorMemberKeys() {
    return findAllMatchedMemberKeys(ClassMemberKey::isConstructor);
  }

  /** Find all member keys based on the given member key predicate. */
  private ImmutableList<ClassMemberKey<?>> findAllMatchedMemberKeys(
      Predicate<ClassMemberKey<?>> predicate) {
    return reasons().keySet().stream().filter(predicate).collect(toImmutableList());
  }

  public final boolean hasTrackingReason(ClassMemberKey<?> classMemberKey) {
    return reasons().containsKey(classMemberKey);
  }

  final boolean hasDeclReason(ClassMemberKey<?> classMemberKey) {
    return hasTrackingReason(classMemberKey) && reasons().get(classMemberKey).hasDeclReason();
  }

  /** Find the original access code for the owner of the class member. */
  final int findOwnerAccessCode(ClassMemberKey<?> memberKey) {
    if (reasons().containsKey(memberKey)) {
      return reasons().get(memberKey).ownerAccess();
    }
    throw new IllegalStateException(String.format("ClassMemberKey Not Found: %s", memberKey));
  }

  /** Find the original access code for the class member declaration. */
  final int findMemberAccessCode(ClassMemberKey<?> memberKey) {
    if (reasons().containsKey(memberKey)) {
      return reasons().get(memberKey).memberAccess();
    }
    throw new IllegalStateException(String.format("ClassMemberKey Not Found: %s", memberKey));
  }

  /** Find all invocation codes of a class member. */
  public final ImmutableList<MemberUseKind> findAllMemberUseKind(ClassMemberKey<?> memberKey) {
    return reasons().containsKey(memberKey)
        ? reasons().get(memberKey).useAccesses().asList()
        : ImmutableList.of();
  }

  @Override
  public ClassMemberRecord acceptTypeMapper(TypeMapper typeMapper) {
    return builder().setReasons(typeMapper.mapKey(reasons())).autoInternalBuild();
  }

  /** The builder for {@link ClassMemberRecord}. */
  @AutoValue.Builder
  public abstract static class ClassMemberRecordBuilder {

    private final Map<ClassMemberKey<?>, ClassMemberTrackReasonBuilder> reasonsCollector =
        new LinkedHashMap<>();

    abstract ClassMemberRecordBuilder setReasons(
        Map<ClassMemberKey<?>, ClassMemberTrackReason> value);

    /** Logs the declaration of a class member. */
    public final ClassMemberTrackReasonBuilder logMemberDecl(
        ClassMemberKey<?> memberKey, int ownerAccess, int memberDeclAccess) {
      return getTrackReason(memberKey).setDeclAccess(ownerAccess, memberDeclAccess);
    }

    /** Logs the use of a class member, including field access and method invocations. */
    public final ClassMemberTrackReasonBuilder logMemberUse(
        ClassMemberKey<?> memberKey, int invokeOpcode) {
      return getTrackReason(memberKey).addUseAccess(invokeOpcode);
    }

    /** Merges an another member record into this record builder. */
    public final ClassMemberRecordBuilder mergeFrom(ClassMemberRecord otherClassMemberRecord) {
      otherClassMemberRecord
          .reasons()
          .forEach(
              (otherMemberKey, otherMemberTrackReason) ->
                  getTrackReason(otherMemberKey).mergeFrom(otherMemberTrackReason));
      return this;
    }

    private ClassMemberTrackReasonBuilder getTrackReason(ClassMemberKey<?> memberKey) {
      return reasonsCollector.computeIfAbsent(
          memberKey, classMemberKey -> ClassMemberTrackReason.builder());
    }

    public final ClassMemberRecord build() {
      return setReasons(
              Maps.transformValues(reasonsCollector, ClassMemberTrackReasonBuilder::build))
          .autoInternalBuild();
    }

    abstract ClassMemberRecord autoInternalBuild();
  }
}
