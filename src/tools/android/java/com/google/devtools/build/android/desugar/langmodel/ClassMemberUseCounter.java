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

import static java.util.stream.Collectors.toCollection;

import com.google.common.collect.ConcurrentHashMultiset;

/** The counter used to track a class member use. */
public final class ClassMemberUseCounter implements TypeMappable<ClassMemberUseCounter> {

  /** Tracks a class member with its associated count. */
  private final ConcurrentHashMultiset<ClassMemberUse> memberUseCounter;

  public ClassMemberUseCounter(ConcurrentHashMultiset<ClassMemberUse> memberUseCounter) {
    this.memberUseCounter = memberUseCounter;
  }

  /** Increases the member use count by one when an member access is encountered. */
  public boolean incrementMemberUseCount(ClassMemberUse classMemberUse) {
    return memberUseCounter.add(classMemberUse);
  }

  /** Retrieves the total use count of a given class member. */
  public long getMemberUseCount(ClassMemberUse memberKey) {
    return memberUseCounter.count(memberKey);
  }

  @Override
  public ClassMemberUseCounter acceptTypeMapper(TypeMapper typeMapper) {
    return new ClassMemberUseCounter(
        memberUseCounter.stream()
            .map(memberUse -> memberUse.acceptTypeMapper(typeMapper))
            .collect(toCollection(ConcurrentHashMultiset::create)));
  }
}
