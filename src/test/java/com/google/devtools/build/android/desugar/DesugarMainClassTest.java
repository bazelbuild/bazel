// Copyright 2016 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.android.desugar.LambdaClassMaker.LAMBDA_METAFACTORY_DUMPER_PROPERTY;
import static org.junit.Assert.fail;

import com.google.common.base.Strings;
import java.io.IOError;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link Desugar} */
@RunWith(JUnit4.class)
public class DesugarMainClassTest {

  @Test
  public void testVerifyLambdaDumpDirectoryRegistration() throws Exception {
    if (Strings.isNullOrEmpty(System.getProperty(LAMBDA_METAFACTORY_DUMPER_PROPERTY))) {
      testLambdaDumpDirSpecifiedInProgramFail();
    } else {
      testLambdaDumpDirPassSpecifiedInCmdPass();
    }
  }

  private void testLambdaDumpDirSpecifiedInProgramFail() throws Exception {
    // This lambda will fail the dump directory registration, which is intended.
    Supplier<Path> supplier =
        () -> {
          Path path = Paths.get(".").toAbsolutePath();
          System.setProperty(LAMBDA_METAFACTORY_DUMPER_PROPERTY, path.toString());
          return path;
        };
    try {
      Desugar.verifyLambdaDumpDirectoryRegistered(supplier.get());
      fail("Expected NullPointerException");
    } catch (NullPointerException e) {
      // Expected
    }
  }

  /**
   * Test the LambdaMetafactory can be correctly set up by specifying the system property {@code
   * LAMBDA_METAFACTORY_DUMPER_PROPERTY} in the command line.
   */
  private void testLambdaDumpDirPassSpecifiedInCmdPass() throws IOException {
    // The following lambda ensures that the LambdaMetafactory is loaded at the beginning of this
    // test, so that the dump directory can be registered.
    Supplier<Path> supplier =
        () -> {
          try {
            return Desugar.createAndRegisterLambdaDumpDirectory();
          } catch (IOException e) {
            throw new IOError(e);
          }
        };
    Path dumpDirectory = supplier.get();
    assertThat(dumpDirectory.toAbsolutePath().toString())
        .isEqualTo(
            Paths.get(System.getProperty(LAMBDA_METAFACTORY_DUMPER_PROPERTY))
                .toAbsolutePath()
                .toString());
    Desugar.verifyLambdaDumpDirectoryRegistered(dumpDirectory);
  }
}
