/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.preanalysis;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.io.FileContentProvider;
import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord.ClassAttributeRecordBuilder;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord.ClassMemberRecordBuilder;
import java.io.IOException;
import java.io.InputStream;
import org.objectweb.asm.Attribute;
import org.objectweb.asm.ClassReader;

/**
 * An analyzer that performs pre-transformation phase analysis and save the analyzed summary to
 * corresponding record builders, such as member references and class attributes.
 */
public class InputPreAnalyzer {

  private final ImmutableList<FileContentProvider<? extends InputStream>> inputFileContents;
  private final Attribute[] customAttributes;

  private final ClassMemberRecordBuilder classMemberRecord;
  private final ClassAttributeRecordBuilder classAttributeRecord;

  public InputPreAnalyzer(
      ImmutableList<FileContentProvider<? extends InputStream>> inputFileContents,
      Attribute[] customAttributes) {
    this.inputFileContents = checkNotNull(inputFileContents);
    this.customAttributes = customAttributes;
    this.classMemberRecord = ClassMemberRecord.builder();
    this.classAttributeRecord = ClassAttributeRecord.builder();
  }

  /** Reads class files and performs class file metadata collection and analysis. */
  public void process() throws IOException {
    for (FileContentProvider<? extends InputStream> inputClassFile : inputFileContents) {
      if (inputClassFile.isClassFile()) {
        try (InputStream inputStream = inputClassFile.get()) {
          ClassReader cr = new ClassReader(inputStream);
          ClassMetadataCollector cv =
              new ClassMetadataCollector(classMemberRecord, classAttributeRecord);
          cr.accept(cv, customAttributes, 0);
        }
      }
    }
  }

  public ClassMemberRecord getClassMemberRecord() {
    return classMemberRecord.build();
  }

  public ClassAttributeRecord getClassAttributeRecord() {
    return classAttributeRecord.build();
  }
}
