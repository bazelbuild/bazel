// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec.local;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.sandbox.SpawnRunnerTestUtil.SpawnExecutionContextForTesting;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Converters.RegexPatternConverter;
import com.google.devtools.common.options.Options;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link AllowedLocalActions}. */
@RunWith(JUnit4.class)
public final class AllowedLocalActionsTest {

  private static final SpawnRunner TEST_RUNNER =
      new SpawnRunner() {
        @Override
        public SpawnResult exec(Spawn spawn, SpawnExecutionContext context) {
          throw new UnsupportedOperationException();
        }

        @Override
        public boolean canExec(Spawn spawn) {
          return true;
        }

        @Override
        public boolean handlesCaching() {
          return false;
        }

        @Override
        public String getName() {
          return "test";
        }
      };

  @Test
  public void denyIfNotAllowed_unsetOptions_returnsNull() throws Exception {
    SpawnExecutionContextForTesting context =
        new SpawnExecutionContextForTesting(/* spawn= */ null, new FileOutErr(), Duration.ZERO);
    LocalExecutionOptions defaultOptions = Options.getDefaults(LocalExecutionOptions.class);

    assertThat(
            AllowedLocalActions.denyIfNotAllowed(defaultOptions, TEST_RUNNER, "Genrule", context))
        .isNull();
  }

  @Test
  public void denyIfNotAllowed_allowedMnemonic_returnsNull() throws Exception {
    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.setAllowedLocalAction(new RegexPatternConverter().convert("Javac"));
    SpawnExecutionContextForTesting context =
        new SpawnExecutionContextForTesting(/* spawn= */ null, new FileOutErr(), Duration.ZERO);

    assertThat(AllowedLocalActions.denyIfNotAllowed(options, TEST_RUNNER, "Javac", context))
        .isNull();
  }

  @Test
  public void denyIfNotAllowed_disallowedMnemonic_flushesBufferedStderrBeforeReturning()
      throws Exception {
    FileSystem bufferedFs =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public synchronized OutputStream getOutputStream(
              PathFragment path, boolean append, boolean internal) throws IOException {
            return new BufferedOutputStream(super.getOutputStream(path, append, internal), 4096);
          }
        };
    assertThat(bufferedFs.getPath("/out").createDirectory()).isTrue();
    FileOutErr fileOutErr =
        new FileOutErr(bufferedFs.getPath("/out/stdout"), bufferedFs.getPath("/out/stderr"));
    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.setAllowedLocalAction(new RegexPatternConverter().convert("Javac"));
    SpawnExecutionContextForTesting context =
        new SpawnExecutionContextForTesting(/* spawn= */ null, fileOutErr, Duration.ZERO);

    SpawnResult result =
        AllowedLocalActions.denyIfNotAllowed(options, TEST_RUNNER, "Genrule", context);

    assertThat(result).isNotNull();
    assertThat(result.status()).isEqualTo(Status.EXECUTION_DENIED);
    assertThat(fileOutErr.hasRecordedStderr()).isTrue();
    assertThat(fileOutErr.errAsLatin1())
        .isEqualTo(
            "Action type Genrule is not allowed to run locally due to regex filter: Javac\n");
  }
}
