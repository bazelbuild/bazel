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

package com.google.devtools.build.android.desugar.nest;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassWriter;

/** Manages the creation and IO stream for nest-companion classes. */
public class NestCompanions {

  private final ClassMemberRecord classMemberRecord;

  /**
   * A map from the class binary names of nest hosts to the associated class writer of the nest's
   * companion.
   */
  private ImmutableMap<String, ClassWriter> companionWriters;

  public static NestCompanions create(ClassMemberRecord classMemberRecord) {
    return new NestCompanions(classMemberRecord);
  }

  private NestCompanions(ClassMemberRecord classMemberRecord) {
    this.classMemberRecord = classMemberRecord;
  }

  /**
   * Generates the nest companion class writers. The nest companion classes will be generated as the
   * last placeholder class type for the synthetic constructor, whose originating constructor has
   * any invocation in other classes in nest.
   */
  ImmutableMap<String, ClassWriter> prepareCompanionClassWriters() {
    return companionWriters =
        classMemberRecord.findAllConstructorMemberKeys().stream()
            .map(ClassMemberKey::nestHost)
            .distinct()
            .collect(
                toImmutableMap(
                    hostName -> hostName, // key
                    hostName -> new ClassWriter(ClassWriter.COMPUTE_MAXS)));
  }

  /**
   * Gets the class visitor of the affiliated nest host of the given class. E.g For the given class
   * com/google/a/b/Delta$Echo, it returns the class visitor of com/google/a/b/Delta$NestCC
   */
  @Nullable
  ClassWriter getCompanionClassWriter(String classInternalName) {
    return companionWriters.get(NestDesugarHelper.nestHost(classInternalName));
  }

  /** Gets all nest companion classes required to be generated. */
  ImmutableList<String> getAllCompanionClasses() {
    return companionWriters.keySet().stream()
        .map(NestDesugarHelper::nestCompanion)
        .collect(toImmutableList());
  }
}
