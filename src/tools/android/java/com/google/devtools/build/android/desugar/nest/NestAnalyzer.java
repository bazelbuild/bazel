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


import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;

/**
 * An analyzer that performs nest-based analysis and save the states to {@link ClassMemberRecord}
 * and generated {@link NestDigest}.
 */
public class NestAnalyzer {

  private NestAnalyzer() {}

  public static NestDigest digest(
      ClassAttributeRecord classAttributeRecord, ClassMemberRecord classMemberRecord) {
    return NestDigest.builder()
        .setClassMemberRecord(classMemberRecord.filterUsedMemberWithTrackedDeclaration())
        .setClassAttributeRecord(classAttributeRecord)
        .build();
  }
}
