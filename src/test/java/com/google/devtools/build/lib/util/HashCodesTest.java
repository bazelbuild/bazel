// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Objects;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class HashCodesTest {
  private static final String O_1 = "one";
  private static final String O_2 = "two";
  private static final String O_3 = "three";
  private static final String O_4 = "four";
  private static final String O_5 = "five";

  @Test
  public void hashObject_returnsHashCode() {
    assertThat(HashCodes.hashObject(O_1)).isEqualTo(O_1.hashCode());
  }

  @Test
  public void hashNull_returnsZero() {
    assertThat(HashCodes.hashObject(null)).isEqualTo(0);
  }

  @Test
  public void hashTwoObjects_sameAsObjectsHash(
      @TestParameter boolean isFirstNull, @TestParameter boolean isSecondNull) {
    Object s1 = isFirstNull ? null : O_1;
    Object s2 = isSecondNull ? null : O_2;
    assertThat(HashCodes.hashObjects(s1, s2)).isEqualTo(Objects.hash(s1, s2));
  }

  @Test
  public void hashThreeObjects_sameAsObjectsHash(
      @TestParameter boolean isFirstNull,
      @TestParameter boolean isSecondNull,
      @TestParameter boolean isThirdNull) {
    Object s1 = isFirstNull ? null : O_1;
    Object s2 = isSecondNull ? null : O_2;
    Object s3 = isThirdNull ? null : O_3;
    assertThat(HashCodes.hashObjects(s1, s2, s3)).isEqualTo(Objects.hash(s1, s2, s3));
  }

  @Test
  public void hashFourObjects_sameAsObjectsHash(
      @TestParameter boolean isFirstNull,
      @TestParameter boolean isSecondNull,
      @TestParameter boolean isThirdNull,
      @TestParameter boolean isFourthNull) {
    Object s1 = isFirstNull ? null : O_1;
    Object s2 = isSecondNull ? null : O_2;
    Object s3 = isThirdNull ? null : O_3;
    Object s4 = isFourthNull ? null : O_4;
    assertThat(HashCodes.hashObjects(s1, s2, s3, s4)).isEqualTo(Objects.hash(s1, s2, s3, s4));
  }

  @Test
  public void hashFiveObjects_sameAsObjectsHash(
      @TestParameter boolean isFirstNull,
      @TestParameter boolean isSecondNull,
      @TestParameter boolean isThirdNull,
      @TestParameter boolean isFourthNull,
      @TestParameter boolean isFifthNull) {
    Object s1 = isFirstNull ? null : O_1;
    Object s2 = isSecondNull ? null : O_2;
    Object s3 = isThirdNull ? null : O_3;
    Object s4 = isFourthNull ? null : O_4;
    Object s5 = isFifthNull ? null : O_5;
    assertThat(HashCodes.hashObjects(s1, s2, s3, s4, s5))
        .isEqualTo(Objects.hash(s1, s2, s3, s4, s5));
  }
}
