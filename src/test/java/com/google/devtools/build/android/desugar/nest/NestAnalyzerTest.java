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
import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.JarEntryRecord;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeJarEntry;
import java.io.IOException;
import java.io.InputStream;
import java.lang.invoke.MethodHandles;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

/** The tests for {@link NestAnalyzer}. */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public final class NestAnalyzerTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addSourceInputsFromJvmFlag("input_srcs")
          .addJavacOptions("-source 11", "-target 11")
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.testsrc.nestanalyzer")
          .enableIterativeTransformation(0)
          .build();

  private final ClassMemberRecord classMemberRecord = ClassMemberRecord.create();
  private final ClassAttributeRecord classAttributeRecord = ClassAttributeRecord.create();
  private final NestCompanions nestCompanions =
      NestCompanions.create(classMemberRecord, classAttributeRecord);

  @Test
  public void emptyInputFiles() throws IOException {
    NestAnalyzer nestAnalyzer =
        new NestAnalyzer(
            ImmutableList.of(), nestCompanions, classMemberRecord, classAttributeRecord);
    nestAnalyzer.analyze();
    assertThat(nestCompanions.getAllCompanionClasses()).isEmpty();
  }

  @Test
  public void companionClassGeneration(
      @RuntimeJarEntry(
              value = "AnalyzedTarget.class",
              round = 0) // Without desugaring at zero-th round.
          JarEntryRecord analyzedTarget)
      throws IOException {

    JarFile jarFile = analyzedTarget.jarFile();
    NestAnalyzer nestAnalyzer =
        new NestAnalyzer(
            jarFile.stream()
                .map(
                    entry ->
                        new FileContentProvider<>(
                            entry.getName(), () -> getJarEntryInputStream(jarFile, entry)))
                .collect(toImmutableList()),
            nestCompanions,
            classMemberRecord,
            classAttributeRecord);

    nestAnalyzer.analyze();

    assertThat(nestCompanions.getAllCompanionClasses())
        .containsExactly(
            "com/google/devtools/build/android/desugar/nest/testsrc/nestanalyzer/AnalyzedTarget$NestCC");
  }

  private static InputStream getJarEntryInputStream(JarFile jarFile, JarEntry entry) {
    try {
      return jarFile.getInputStream(entry);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
}
