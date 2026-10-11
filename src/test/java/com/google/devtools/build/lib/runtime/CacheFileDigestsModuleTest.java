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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.runtime.commands.CqueryCommand;
import com.google.devtools.build.lib.runtime.commands.InfoCommand;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for digest caching before the execution tool is initialized. */
@RunWith(JUnit4.class)
public final class CacheFileDigestsModuleTest {
  private final AtomicInteger reads = new AtomicInteger();
  private final FileSystem fileSystem =
      new InMemoryFileSystem(DigestHashFunction.SHA256) {
        @Override
        public byte[] getDigest(PathFragment path) throws IOException {
          reads.incrementAndGet();
          return super.getDigest(path);
        }
      };
  private final Path file = fileSystem.getPath("/watched-input");

  @After
  public void resetCache() {
    DigestUtils.configureCache(0);
  }

  @Test
  public void initializesAndPreservesCacheBeforeLoading() throws Exception {
    FileSystemUtils.writeContentAsLatin1(file, "repository input");
    OptionsParser buildOptions =
        OptionsParser.builder().optionsClasses(ExecutionOptions.class).build();
    buildOptions.parse("--cache_computed_file_digests=100");
    OptionsParser infoOptions =
        OptionsParser.builder().optionsClasses(ExecutionOptions.class).build();
    CommandEnvironment env = mock(CommandEnvironment.class);
    CacheFileDigestsModule module = new CacheFileDigestsModule();

    // Info inherits execution options but must not resize a previous build's cache.
    for (var command : List.of(CqueryCommand.class, InfoCommand.class, BuildCommand.class)) {
      when(env.getCommand()).thenReturn(command.getAnnotation(Command.class));
      when(env.getOptions()).thenReturn(command == InfoCommand.class ? infoOptions : buildOptions);
      module.beforeCommand(env);
      byte[] digest = DigestUtils.manuallyComputeDigest(file, null);
      assertThat(DigestUtils.manuallyComputeDigest(file, null)).isEqualTo(digest);
      assertThat(reads.get()).isEqualTo(1);
      module.commandComplete();
    }
  }

  @Test
  public void zeroSizeDisablesPreviousCacheBeforeLoading() throws Exception {
    FileSystemUtils.writeContentAsLatin1(file, "repository input");
    DigestUtils.configureCache(100);
    byte[] digest = DigestUtils.manuallyComputeDigest(file, null);
    OptionsParser options = OptionsParser.builder().optionsClasses(ExecutionOptions.class).build();
    options.parse("--cache_computed_file_digests=0");
    CommandEnvironment env = mock(CommandEnvironment.class);
    when(env.getCommand()).thenReturn(BuildCommand.class.getAnnotation(Command.class));
    when(env.getOptions()).thenReturn(options);

    CacheFileDigestsModule module = new CacheFileDigestsModule();
    module.beforeCommand(env);
    assertThat(DigestUtils.manuallyComputeDigest(file, null)).isEqualTo(digest);
    assertThat(DigestUtils.manuallyComputeDigest(file, null)).isEqualTo(digest);
    assertThat(reads.get()).isEqualTo(3);
    module.commandComplete();
  }
}
