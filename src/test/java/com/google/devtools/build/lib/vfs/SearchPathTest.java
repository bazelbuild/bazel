// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.OutputStream;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SearchPathTest {
  private FileSystem fs = new InMemoryFileSystem();

  @Test
  public void testNull() throws Exception {
    assertThat(SearchPath.parse(fs, null)).isEmpty();
  }

  @Test
  public void testBasic() throws Exception {
    fs.getPath("/bin").createDirectory();
    List<Path> searchPath = ImmutableList.of(fs.getPath("/"), fs.getPath("/bin"));
    assertThat(SearchPath.parse(fs, "/:/bin")).isEqualTo(searchPath);
    assertThat(SearchPath.parse(fs, ".:/:/bin")).isEqualTo(searchPath);

    try (OutputStream out = fs.getOutputStream(fs.getPath("/bin/exe"))) {
      out.write(new byte[5]);
    }

    assertThat(SearchPath.which(searchPath, "exe")).isNull();

    fs.getPath("/bin/exe").setExecutable(true);
    assertThat(SearchPath.which(searchPath, "exe")).isEqualTo(fs.getPath("/bin/exe"));

    assertThat(SearchPath.which(searchPath, "bin/exe")).isNull();
    assertThat(SearchPath.which(searchPath, "/bin/exe")).isNull();
  }
}
