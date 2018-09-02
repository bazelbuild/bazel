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

package com.google.devtools.build.importdeps;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import java.nio.file.Path;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class MainTest {

  @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

  @Ignore // TODO(cushon): re-enable after cl/210237269
  @Test
  public void usage() throws Exception {
    Path lib = tempFolder.newFile("lib.jar").toPath();
    Path in = tempFolder.newFile("in.jar").toPath();
    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Main.parseCommandLineOptions(
                    new String[] {
                      "--bootclasspath_entry",
                      lib.toString(),
                      "--classpath_entry",
                      lib.toString(),
                      "--input",
                      in.toString()
                    }));
    assertThat(thrown).hasMessageThat().contains("--jdeps_output");
  }
}
