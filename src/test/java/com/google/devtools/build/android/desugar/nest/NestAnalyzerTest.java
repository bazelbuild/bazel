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
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.io.FileContentProvider;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;
import com.google.devtools.build.runtime.RunfilesPaths;
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** The tests for {@link NestAnalyzer}. */
@RunWith(JUnit4.class)
@MediumTest(MediumTestAttribute.FILE)
public class NestAnalyzerTest {

  private final ClassMemberRecord classMemberRecord = ClassMemberRecord.create();
  private final NestCompanions nestCompanions = NestCompanions.create(classMemberRecord);

  @Test
  public void emptyInputFiles() throws IOException {
    NestAnalyzer nestAnalyzer =
        new NestAnalyzer(ImmutableList.of(), nestCompanions, classMemberRecord);
    nestAnalyzer.analyze();
    assertThat(nestCompanions.getAllCompanionClasses()).isEmpty();
  }

  @Test
  public void companionClassGeneration() throws IOException {
    ZipFile jarFile =
        new ZipFile(
            RunfilesPaths.resolve(
                    "third_party/bazel/src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/nestanalyzer/libanalyzed_target.jar")
                .toFile());
    NestAnalyzer nestAnalyzer =
        new NestAnalyzer(
            jarFile.stream()
                .map(
                    entry ->
                        new FileContentProvider<>(
                            entry.getName(), () -> getZipEntryInputStream(jarFile, entry)))
                .collect(toImmutableList()),
            nestCompanions,
            classMemberRecord);

    nestAnalyzer.analyze();

    assertThat(nestCompanions.getAllCompanionClasses())
        .containsExactly(
            "com/google/devtools/build/android/desugar/nest/testsrc/nestanalyzer/AnalyzedTarget$NestCC");
  }

  private static InputStream getZipEntryInputStream(ZipFile jarFile, ZipEntry entry) {
    try {
      return jarFile.getInputStream(entry);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
}
