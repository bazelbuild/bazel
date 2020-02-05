// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.getStrategyClassName;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.getTwrStrategyClassNameSpecifiedInSystemProperty;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.isMimicStrategy;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.isNullStrategy;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.isReuseStrategy;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.junit.Assert.fail;

import com.google.devtools.build.android.desugar.runtime.ThrowableExtension;
import com.google.devtools.build.android.desugar.testdata.ClassUsingTryWithResources;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** The functional test for desugaring try-with-resources. */
@RunWith(JUnit4.class)
public class DesugarTryWithResourcesFunctionalTest {

  @Test
  public void testCheckSuppressedExceptionsReturningEmptySuppressedExceptions() {
    Throwable[] suppressed = ClassUsingTryWithResources.checkSuppressedExceptions(false);
    assertThat(suppressed).isEmpty();
  }

  @Test
  public void testPrintStackTraceOfCaughtException() {
    String trace = ClassUsingTryWithResources.printStackTraceOfCaughtException();
    if (isMimicStrategy()) {
      assertThat(trace.toLowerCase()).contains("suppressed");
    } else if (isReuseStrategy()) {
      assertThat(trace.toLowerCase()).contains("suppressed");
    } else if (isNullStrategy()) {
      assertThat(trace.toLowerCase()).doesNotContain("suppressed");
    } else {
      fail("unexpected desugaring strategy " + ThrowableExtension.getStrategy());
    }
  }

  @Test
  public void testCheckSuppressedExceptionReturningOneSuppressedException() {
    Throwable[] suppressed = ClassUsingTryWithResources.checkSuppressedExceptions(true);

    if (isMimicStrategy()) {
      assertThat(suppressed).hasLength(1);
    } else if (isReuseStrategy()) {
      assertThat(suppressed).hasLength(1);
    } else if (isNullStrategy()) {
      assertThat(suppressed).isEmpty();
    } else {
      fail("unexpected desugaring strategy " + ThrowableExtension.getStrategy());
    }
  }

  @Test
  public void testSimpleTryWithResources() {

    Exception expected =
        assertThrows(Exception.class, () -> ClassUsingTryWithResources.simpleTryWithResources());
    assertThat(expected.getClass()).isEqualTo(RuntimeException.class);

    String expectedStrategyName = getTwrStrategyClassNameSpecifiedInSystemProperty();
    assertThat(getStrategyClassName()).isEqualTo(expectedStrategyName);
    if (isMimicStrategy()) {
      assertThat(expected.getSuppressed()).isEmpty();
      assertThat(ThrowableExtension.getSuppressed(expected)).hasLength(1);
      assertThat(ThrowableExtension.getSuppressed(expected)[0].getClass())
          .isEqualTo(IOException.class);
    } else if (isReuseStrategy()) {
      assertThat(expected.getSuppressed()).hasLength(1);
      assertThat(expected.getSuppressed()[0].getClass()).isEqualTo(IOException.class);
      assertThat(ThrowableExtension.getSuppressed(expected)[0].getClass())
          .isEqualTo(IOException.class);
    } else if (isNullStrategy()) {
      assertThat(expected.getSuppressed()).isEmpty();
      assertThat(ThrowableExtension.getSuppressed(expected)).isEmpty();
    } else {
      fail("unexpected desugaring strategy " + getStrategyClassName());
    }
  }

  @Test
  public void simpleTryWithResources_multipleClosableResources() {

    IOException expected =
        assertThrows(
            IOException.class, () -> ClassUsingTryWithResources.multipleTryWithResources());

    String expectedStrategyName = getTwrStrategyClassNameSpecifiedInSystemProperty();
    assertThat(getStrategyClassName()).isEqualTo(expectedStrategyName);
    if (isMimicStrategy()) {
      assertThat(expected.getSuppressed()).isEmpty();
      assertThat(ThrowableExtension.getSuppressed(expected)).hasLength(2);
      assertThat(ThrowableExtension.getSuppressed(expected)[0].getClass())
          .isEqualTo(IOException.class);
    } else if (isReuseStrategy()) {
      assertThat(expected.getSuppressed()).hasLength(2);
      assertThat(expected.getSuppressed()[0].getClass()).isEqualTo(IOException.class);
      assertThat(ThrowableExtension.getSuppressed(expected)[0].getClass())
          .isEqualTo(IOException.class);
    } else if (isNullStrategy()) {
      assertThat(expected.getSuppressed()).isEmpty();
      assertThat(ThrowableExtension.getSuppressed(expected)).isEmpty();
    } else {
      fail("unexpected desugaring strategy " + getStrategyClassName());
    }
  }

  @Test
  public void testInheritanceTryWithResources() {

    Exception expected =
        assertThrows(
            Exception.class, () -> ClassUsingTryWithResources.inheritanceTryWithResources());
    assertThat(expected.getClass()).isEqualTo(RuntimeException.class);

    String expectedStrategyName = getTwrStrategyClassNameSpecifiedInSystemProperty();
    assertThat(getStrategyClassName()).isEqualTo(expectedStrategyName);
    if (isMimicStrategy()) {
      assertThat(expected.getSuppressed()).isEmpty();
      assertThat(ThrowableExtension.getSuppressed(expected)).hasLength(1);
      assertThat(ThrowableExtension.getSuppressed(expected)[0].getClass())
          .isEqualTo(IOException.class);
    } else if (isReuseStrategy()) {
      assertThat(expected.getSuppressed()).hasLength(1);
      assertThat(expected.getSuppressed()[0].getClass()).isEqualTo(IOException.class);
      assertThat(ThrowableExtension.getSuppressed(expected)[0].getClass())
          .isEqualTo(IOException.class);
    } else if (isNullStrategy()) {
      assertThat(expected.getSuppressed()).isEmpty();
      assertThat(ThrowableExtension.getSuppressed(expected)).isEmpty();
    } else {
      fail("unexpected desugaring strategy " + getStrategyClassName());
    }
  }
}
