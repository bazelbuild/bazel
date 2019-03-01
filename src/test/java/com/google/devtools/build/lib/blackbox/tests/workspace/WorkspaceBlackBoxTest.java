// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.tests.workspace;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.Test;

/** End to end test of workspace-related functionality. */
public class WorkspaceBlackBoxTest extends AbstractBlackBoxTest {

  @Test
  public void testWorkspaceChanges() throws Exception {
    Path repoA = context().getTmpDir().resolve("a");
    new RepoWithRuleWritingTextGenerator(repoA).withOutputText("hi").setupRepository();

    Path repoB = context().getTmpDir().resolve("b");
    new RepoWithRuleWritingTextGenerator(repoB).withOutputText("bye").setupRepository();

    context()
        .write(
            WORKSPACE,
            String.format(
                "local_repository(name = 'x', path = '%s',)",
                PathUtils.pathForStarlarkFile(repoA)));
    context().bazel().build("@x//:" + RepoWithRuleWritingTextGenerator.TARGET);

    Path xPath = context().resolveBinPath(context().bazel(), "external/x/out");
    assertThat(Files.exists(xPath)).isTrue();
    List<String> lines = PathUtils.readFile(xPath);
    assertThat(lines.size()).isEqualTo(1);
    assertThat(lines.get(0)).isEqualTo("hi");

    context()
        .write(
            WORKSPACE,
            String.format(
                "local_repository(name = 'x', path = '%s',)",
                PathUtils.pathForStarlarkFile(repoB)));
    context().bazel().build("@x//:" + RepoWithRuleWritingTextGenerator.TARGET);

    assertThat(Files.exists(xPath)).isTrue();
    lines = PathUtils.readFile(xPath);
    assertThat(lines.size()).isEqualTo(1);
    assertThat(lines.get(0)).isEqualTo("bye");
  }

  @Test
  public void testPathWithSpace() throws Exception {
    context().write("a b/WORKSPACE");
    context().bazel().info();
    context().bazel().help();
  }

  // TODO(ichern) move other tests from workspace_test.sh here.

}
