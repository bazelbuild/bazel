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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.writeIsoLatin1;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.packages.util.SubincludePreprocessor;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.nio.CharBuffer;
import java.nio.charset.StandardCharsets;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class SubincludePreprocessorTest extends PackageLoadingTestCase {
  private Path packageRoot;
  protected SubincludePreprocessor preprocessor;

  public SubincludePreprocessorTest() {}

  @Before
  public final void createPreprocessor() throws Exception {
    preprocessor = new SubincludePreprocessor(scratch.getFileSystem(), getPackageManager());
    packageRoot = rootDirectory.getChild("preprocessing");
    assertTrue(packageRoot.createDirectory());
    reporter.removeHandler(failFastHandler);
  }

  @After
  public final void resetPreprocessor() throws Exception {
    preprocessor = null;
  }

  private ParserInputSource createInputSource(String... lines) throws Exception {
    Path buildFile = packageRoot.getChild("BUILD");
    writeIsoLatin1(buildFile, lines);
    ParserInputSource in = ParserInputSource.create(buildFile);
    return in;
  }

  protected Preprocessor.Result preprocess(ParserInputSource in, String packageName)
      throws IOException, InterruptedException {
    Path buildFilePath = packageRoot.getRelative(in.getPath());
    byte[] buildFileBytes =
        StandardCharsets.ISO_8859_1.encode(CharBuffer.wrap(in.getContent())).array();
    Preprocessor.Result result =
        preprocessor.preprocess(
            buildFilePath,
            buildFileBytes,
            packageName, /*globber=*/
            null,
            /*ruleNames=*/ null);
    Event.replayEventsOn(reporter, result.events);
    return result;
  }

  @Test
  public void testPreprocessingInclude() throws Exception {
    ParserInputSource in = createInputSource("subinclude('//foo:bar')");

    scratch.file("foo/BUILD");
    scratch.file("foo/bar", "genrule('turtle1')", "subinclude('//foo:baz')");
    scratch.file("foo/baz", "genrule('turtle2')");

    String out = assertPreprocessingSucceeds(in);
    assertThat(out).containsMatch("turtle1");
    assertThat(out).containsMatch("turtle2");
    assertThat(out).containsMatch("mocksubinclude\\('//foo:bar', *'/workspace/foo/bar'\\)");
    assertThat(out).containsMatch("mocksubinclude\\('//foo:baz', *'/workspace/foo/baz'\\)");
  }

  @Test
  public void testSubincludeNotFound() throws Exception {
    ParserInputSource in = createInputSource("subinclude('//nonexistent:bar')");
    scratch.file("foo/BUILD");
    String out = assertPreprocessingSucceeds(in);
    assertThat(out).containsMatch("mocksubinclude\\('//nonexistent:bar', *''\\)");
    assertContainsEvent("Cannot find subincluded file");
  }

  @Test
  public void testError() throws Exception {
    ParserInputSource in = createInputSource("subinclude('//foo:bar')");
    scratch.file("foo/BUILD");
    scratch.file("foo/bar", SubincludePreprocessor.TRANSIENT_ERROR);
    try {
      preprocess(in, "path/to/package");
      fail();
    } catch (IOException e) {
      // Expected.
    }
  }

  public String assertPreprocessingSucceeds(ParserInputSource in) throws Exception {
    Preprocessor.Result out = preprocess(in, "path/to/package");
    assertTrue(out.preprocessed);

    // Check that the preprocessed file looks plausible:
    assertEquals(in.getPath(), out.result.getPath());
    String outString = new String(out.result.getContent());
    outString = outString.replaceAll("[\n \t]", ""); // for easier regexps
    return outString;
  }
}
