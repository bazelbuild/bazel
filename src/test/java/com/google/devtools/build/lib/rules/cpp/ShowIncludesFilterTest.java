// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.ByteArrayOutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link ShowIncludesFilter}. */
@RunWith(JUnit4.class)
public class ShowIncludesFilterTest {

  private ShowIncludesFilter showIncludesFilter;
  private ByteArrayOutputStream output;
  private FilterOutputStream filterOutputStream;
  private FileSystem fs;
  private FileOutErr outErr;

  @Before
  public void setUpOutputStreams() throws IOException {
    showIncludesFilter = new ShowIncludesFilter("foo.cpp");
    output = new ByteArrayOutputStream();
    filterOutputStream = showIncludesFilter.getFilteredOutputStream(output);
    fs = new InMemoryFileSystem();
    fs.getRootDirectory().getChild("out").createDirectory();
  }

  private byte[] getBytes(String str) {
    return str.getBytes(UTF_8);
  }

  @Test
  public void testNotMatch() throws IOException {
    // Normal output message with newline
    filterOutputStream.write(getBytes("I am compiling\n"));
    assertThat(output.toString()).isEqualTo("I am compiling\n");
  }

  @Test
  public void testNotMatchThenFlushing() throws IOException {
    // Normal output message without newline
    filterOutputStream.write(getBytes("Still compiling"));
    assertThat(output.toString()).isEmpty();
    filterOutputStream.flush();
    // flush to output should succeed
    assertThat(output.toString()).isEqualTo("Still compiling");
  }

  @Test
  public void testMatchPartOfNotePrefix() throws IOException {
    // Prefix of "Note: including file:"
    filterOutputStream.write(getBytes("Note: "));
    filterOutputStream.flush();
    // flush to output shouldn't work, because there's still a chance to match.
    assertThat(output.toString()).isEmpty();
    // "Note: other info" doesn't match "Note: including file:", it's ok to flush.
    filterOutputStream.write(getBytes("other info"));
    filterOutputStream.flush();
    assertThat(output.toString()).isEqualTo("Note: other info");
  }

  @Test
  public void testMatchAllOfNotePrefix() throws IOException {
    // "Note: including file:" is the prefix
    filterOutputStream.write(getBytes("Note: including file: bar.h"));
    filterOutputStream.flush();
    // flush to output should not work, waiting for newline
    assertThat(output.toString()).isEmpty();
    filterOutputStream.write(getBytes("\n"));
    // It's a match, output should be filtered, dependency on bar.h should be found.
    assertThat(output.toString()).isEmpty();
    assertThat(showIncludesFilter.getDependencies()).contains("bar.h");
  }

  @Test
  public void testMatchSourceFileName() throws IOException {
    filterOutputStream.write(getBytes("foo.cpp\n"));
    // It's a match, output should be filtered, no dependency found.
    assertThat(output.toString()).isEmpty();
    assertThat(showIncludesFilter.getDependencies()).isEmpty();
  }

  @Test
  public void testMatchPartOfSourceFileName() throws IOException {
    filterOutputStream.write(getBytes("foo"));
    filterOutputStream.flush();
    assertThat(output.toString()).isEmpty();

    filterOutputStream.write(getBytes(".h"));
    filterOutputStream.flush();
    assertThat(output.toString()).isEqualTo("foo.h");
  }

  @Test
  public void testOnFileOutErr() throws IOException {
    outErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    ShowIncludesFilter showIncludesFilterForStdout = new ShowIncludesFilter("foo.cpp");
    ShowIncludesFilter showIncludesFilterForStderr = new ShowIncludesFilter("foo.cpp");
    outErr.setOutputFilter(showIncludesFilterForStdout);
    outErr.setErrorFilter(showIncludesFilterForStderr);
    outErr.getOutputStream().write(getBytes("Note: including file: bar1.h\n"));
    outErr.getErrorStream().write(getBytes("Note: including file: bar2.h\n"));
    assertThat(showIncludesFilterForStdout.getDependencies()).contains("bar1.h");
    assertThat(showIncludesFilterForStderr.getDependencies()).contains("bar2.h");
  }
}
