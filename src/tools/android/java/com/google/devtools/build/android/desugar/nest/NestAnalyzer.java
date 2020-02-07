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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.io.FileContentProvider;
import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;
import java.io.IOException;
import java.io.InputStream;
import org.objectweb.asm.ClassReader;

/**
 * An analyzer that performs nest-based analysis and save the states to {@link ClassMemberRecord}
 * and generated {@link NestDigest}.
 */
public class NestAnalyzer {

  private final ImmutableList<FileContentProvider<? extends InputStream>> inputFileContents;
  private final NestDigest nestDigest;
  private final ClassMemberRecord classMemberRecord;
  private final ClassAttributeRecord classAttributeRecord;

  /**
   * Perform a nest-based analysis of input classes, including tracking private member access
   * outside its owner.
   *
   * @return A manager class for nest companions.
   */
  public static NestDigest analyzeNests(
      ImmutableList<FileContentProvider<? extends InputStream>> inputFileContents,
      ClassMemberRecord classMemberRecord,
      ClassAttributeRecord classAttributeRecord)
      throws IOException {
    NestDigest nestDigest = NestDigest.create(classMemberRecord, classAttributeRecord);
    NestAnalyzer nestAnalyzer =
        new NestAnalyzer(inputFileContents, nestDigest, classMemberRecord, classAttributeRecord);
    nestAnalyzer.analyze();
    return nestDigest;
  }

  @VisibleForTesting
  NestAnalyzer(
      ImmutableList<FileContentProvider<? extends InputStream>> inputFileContents,
      NestDigest nestDigest,
      ClassMemberRecord classMemberRecord,
      ClassAttributeRecord classAttributeRecord) {
    this.inputFileContents = checkNotNull(inputFileContents);
    this.nestDigest = checkNotNull(nestDigest);
    this.classMemberRecord = checkNotNull(classMemberRecord);
    this.classAttributeRecord = classAttributeRecord;
  }

  /** Performs class member declaration and usage analysis of files. */
  @VisibleForTesting
  void analyze() throws IOException {
    for (FileContentProvider<? extends InputStream> inputClassFile : inputFileContents) {
      if (inputClassFile.isClassFile()) {
        try (InputStream inputStream = inputClassFile.get()) {
          ClassReader cr = new ClassReader(inputStream);
          CrossMateMainCollector cv =
              new CrossMateMainCollector(classMemberRecord, classAttributeRecord);
          cr.accept(cv, 0);
        }
      }
    }
    classMemberRecord.filterUsedMemberWithTrackedDeclaration();
    nestDigest.prepareCompanionClasses();
  }
}
