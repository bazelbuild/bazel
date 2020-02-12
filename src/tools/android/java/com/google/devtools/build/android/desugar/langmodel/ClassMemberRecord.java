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

import com.google.common.collect.ImmutableList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Predicate;

/**
 * A record that tracks the declarations and usages of a class member, including fields,
 * constructors and methods.
 */
public final class ClassMemberRecord implements TypeMappable<ClassMemberRecord> {

  /** Tracks a class member with a reason. */
  private final Map<ClassMemberKey<?>, ClassMemberTrackReason> reasons;

  /** The factory method of this class. */
  public static ClassMemberRecord create() {
    return new ClassMemberRecord(new LinkedHashMap<>());
  }

  /** Gets class members with both tracked declarations and tracked usage. */
  public boolean filterUsedMemberWithTrackedDeclaration() {
    return reasons
        .values()
        .removeIf(
            reason -> {
              // keep interface members and class members with use references.
              return !(reason.hasInterfaceDeclReason()
                  || (reason.hasDeclReason() && reason.hasMemberUseReason()));
            });
  }

  private ClassMemberRecord(Map<ClassMemberKey<?>, ClassMemberTrackReason> reasons) {
    this.reasons = reasons;
  }

  /** Find all member keys that represent a constructor. */
  public ImmutableList<ClassMemberKey<?>> findAllConstructorMemberKeys() {
    return findAllMatchedMemberKeys(ClassMemberKey::isConstructor);
  }

  /** Find all member keys based on the given member key predicate. */
  ImmutableList<ClassMemberKey<?>> findAllMatchedMemberKeys(
      Predicate<ClassMemberKey<?>> predicate) {
    return reasons.keySet().stream().filter(predicate).collect(toImmutableList());
  }

  /** Whether this record has any reason to desugar a nest. */
  public boolean hasAnyReason() {
    return !reasons.isEmpty();
  }

  public boolean hasTrackingReason(ClassMemberKey<?> classMemberKey) {
    return reasons.containsKey(classMemberKey);
  }

  boolean hasDeclReason(ClassMemberKey<?> classMemberKey) {
    return hasTrackingReason(classMemberKey) && reasons.get(classMemberKey).hasDeclReason();
  }

  /** Find the original access code for the owner of the class member. */
  int findOwnerAccessCode(ClassMemberKey<?> memberKey) {
    if (reasons.containsKey(memberKey)) {
      return reasons.get(memberKey).getOwnerAccess();
    }
    throw new IllegalStateException(String.format("ClassMemberKey Not Found: %s", memberKey));
  }

  /** Find the original access code for the class member declaration. */
  int findMemberAccessCode(ClassMemberKey<?> memberKey) {
    if (reasons.containsKey(memberKey)) {
      return reasons.get(memberKey).getMemberAccess();
    }
    throw new IllegalStateException(String.format("ClassMemberKey Not Found: %s", memberKey));
  }

  /** Find all invocation codes of a class member. */
  public ImmutableList<MemberUseKind> findAllMemberUseKind(ClassMemberKey<?> memberKey) {
    if (reasons.containsKey(memberKey)) {
      return ImmutableList.copyOf(reasons.get(memberKey).getUseAccesses());
    }
    return ImmutableList.of();
  }

  /** Logs the declaration of a class member. */
  public ClassMemberTrackReason logMemberDecl(
      ClassMemberKey<?> memberKey, int ownerAccess, int memberDeclAccess) {
    return reasons
        .computeIfAbsent(memberKey, classMemberKey -> new ClassMemberTrackReason())
        .setDeclAccess(ownerAccess, memberDeclAccess);
  }

  /** Logs the use of a class member, including field access and method invocations. */
  public ClassMemberTrackReason logMemberUse(ClassMemberKey<?> memberKey, int invokeOpcode) {
    return reasons
        .computeIfAbsent(memberKey, classMemberKey -> new ClassMemberTrackReason())
        .addUseAccess(invokeOpcode);
  }

  /** Merge an another member record into this record. */
  public void mergeFrom(ClassMemberRecord otherClassMemberRecord) {
    otherClassMemberRecord.reasons.forEach(
        (classMemberKey, classMemberTrackReason) ->
            reasons.merge(
                classMemberKey, classMemberTrackReason, ClassMemberTrackReason::mergeFrom));
  }

  @Override
  public ClassMemberRecord acceptTypeMapper(TypeMapper typeMapper) {
    return new ClassMemberRecord(typeMapper.mapKey(reasons));
  }
}
