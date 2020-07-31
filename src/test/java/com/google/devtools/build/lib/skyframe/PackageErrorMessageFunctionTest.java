// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.skyframe.PackageErrorMessageValue.Result;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PackageErrorMessageFunction}. */
@RunWith(JUnit4.class)
public class PackageErrorMessageFunctionTest extends BuildViewTestCase {

  @Test
  public void testNoErrorMessage() throws Exception {
    scratch.file("a/BUILD");
    assertThat(getPackageErrorMessageValue(/*keepGoing=*/ false).getResult())
        .isEqualTo(Result.NO_ERROR);
  }

  @Test
  public void testPackageWithErrors() throws Exception {
    // Opt out of failing fast on an error event.
    reporter.removeHandler(failFastHandler);

    scratch.file("a/BUILD", "imaginary_macro(name = 'this macro is not defined')");

    assertThat(getPackageErrorMessageValue(/*keepGoing=*/ false).getResult())
        .isEqualTo(Result.ERROR);
  }

  @Test
  public void testNoSuchPackageException() throws Exception {
    scratch.file("a/BUILD", "load('//a:does_not_exist.bzl', 'imaginary_macro')");

    PackageErrorMessageValue packageErrorMessageValue =
        getPackageErrorMessageValue(/*keepGoing=*/ true);
    assertThat(packageErrorMessageValue.getResult()).isEqualTo(Result.NO_SUCH_PACKAGE_EXCEPTION);
    assertThat(packageErrorMessageValue.getNoSuchPackageExceptionMessage())
        .isEqualTo("error loading package 'a': cannot load '//a:does_not_exist.bzl': no such file");
  }

  private PackageErrorMessageValue getPackageErrorMessageValue(boolean keepGoing)
      throws InterruptedException {
    SkyKey key = PackageErrorMessageValue.key(PackageIdentifier.createInMainRepo("a"));
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(keepGoing)
            .setNumThreads(SequencedSkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHandler(reporter)
            .build();
    EvaluationResult<SkyValue> result =
        skyframeExecutor.getDriver().evaluate(ImmutableList.of(key), evaluationContext);
    assertThat(result.hasError()).isFalse();
    SkyValue value = result.get(key);
    assertThat(value).isInstanceOf(PackageErrorMessageValue.class);
    return (PackageErrorMessageValue) value;
  }
}
