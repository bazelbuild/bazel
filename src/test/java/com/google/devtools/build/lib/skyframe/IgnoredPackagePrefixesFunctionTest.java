// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link IgnoredPackagePrefixesFunction}. */
@RunWith(JUnit4.class)
public class IgnoredPackagePrefixesFunctionTest extends BuildViewTestCase {

  private IgnoredPackagePrefixesValue executeFunction(SkyKey key) throws Exception {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    EvaluationResult<IgnoredPackagePrefixesValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /*keepGoing=*/ false, reporter);
    if (result.hasError()) {
      throw result.getError(key).getException();
    }
    return result.get(key);
  }

  @Test
  public void main_noIgnore() throws Exception {
    scratch.overwriteFile(".bazelignore");
    IgnoredPackagePrefixesValue result = executeFunction(
        IgnoredPackagePrefixesValue.key());
    assertThat(result).isNotNull();
    assertThat(result.getPatterns()).isEmpty();
  }

  @Test
  public void main_ignore() throws Exception {
    scratch.overwriteFile(".bazelignore", "foo");
    IgnoredPackagePrefixesValue result = executeFunction(
        IgnoredPackagePrefixesValue.key());
    assertThat(result).isNotNull();
    assertThat(result.getPatterns()).containsExactly(PathFragment.create("foo"));
  }

  @Test
  public void repository_noIgnore() throws Exception {
    scratch.dir("repo");
    scratch.file("repo/WORKSPACE");
    scratch.file("repo/BUILD");
    scratch.file("repo/.bazelignore");
    rewriteWorkspace("local_repository(name = 'repo', path = 'repo')");

    IgnoredPackagePrefixesValue result = executeFunction(
        IgnoredPackagePrefixesValue.key(RepositoryName.create("@repo")));
    assertThat(result).isNotNull();
    assertThat(result.getPatterns()).isEmpty();
  }

  @Test
  public void repository_ignore() throws Exception {
    scratch.dir("repo");
    scratch.file("repo/WORKSPACE");
    scratch.file("repo/BUILD");
    scratch.file("repo/.bazelignore", "foo");
    rewriteWorkspace("local_repository(name = 'repo', path = 'repo')");

    IgnoredPackagePrefixesValue result = executeFunction(
        IgnoredPackagePrefixesValue.key(RepositoryName.create("@repo")));
    assertThat(result).isNotNull();
    assertThat(result.getPatterns()).containsExactly(PathFragment.create("foo"));
  }

  @Test
  public void repository_error() throws Exception {
    scratch.deleteFile("repo");
    // This should yield an error.
    rewriteWorkspace("local_repository(name = 'repo', path = 'repo')");

    assertThrows(
        IOException.class,
        () -> executeFunction(
            IgnoredPackagePrefixesValue.key(RepositoryName.create("@repo"))));
  }
}
