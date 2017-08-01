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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.ExistingPathStringDictionaryConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.PathListConverter;
import com.google.devtools.build.android.Converters.StringDictionaryConverter;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link Converters}.
 */
@RunWith(JUnit4.class)
public final class ConvertersTest {

  private static final String SEPARATOR = File.pathSeparator;

  @Rule public final TemporaryFolder tmp = new TemporaryFolder();

  @Rule
  public final ExpectedException expected = ExpectedException.none();

  @Test
  public void testPathConverter_empty() throws Exception {
    PathConverter converter = new PathConverter();
    Path result = converter.convert("");
    assertThat((Object) result).isEqualTo(Paths.get(""));
  }

  @Test
  public void testPathConverter_invalid() throws Exception {
    String arg = "\u0000";
    PathConverter converter = new PathConverter();
    expected.expect(OptionsParsingException.class);
    expected.expectMessage(String.format("%s is not a valid path:", arg));
    converter.convert(arg);
  }

  @Test
  public void testPathConverter_valid() throws Exception {
    PathConverter converter = new PathConverter();
    Path result = converter.convert("test_file");
    assertThat((Object) result).isEqualTo(Paths.get("test_file"));
  }

  @Test
  public void testExistingPathConverter_nonExisting() throws Exception {
    String arg = "test_file";
    ExistingPathConverter converter = new ExistingPathConverter();
    expected.expect(OptionsParsingException.class);
    expected.expectMessage(String.format("%s is not a valid path: it does not exist.", arg));
    converter.convert(arg);
  }

  @Test
  public void testExistingPathConverter_existing() throws Exception {
    Path testFile = tmp.newFile("test_file").toPath();
    ExistingPathConverter converter = new ExistingPathConverter();
    Path result = converter.convert(testFile.toString());
    assertThat((Object) result).isEqualTo(testFile);
  }

  @Test
  public void testPathListConverter() throws Exception {
    PathListConverter converter = new PathListConverter();
    List<Path> result =
        converter.convert("foo" + SEPARATOR + "bar" + SEPARATOR + SEPARATOR + "baz" + SEPARATOR);
    assertThat(result)
        .containsAllOf(Paths.get("foo"), Paths.get("bar"), Paths.get("baz"))
        .inOrder();
  }

  @Test
  public void testStringDictionaryConverter_emptyEntry() throws Exception {
    StringDictionaryConverter converter = new StringDictionaryConverter();
    expected.expect(OptionsParsingException.class);
    expected.expectMessage("Dictionary entry [] does not contain both a key and a value.");
    converter.convert("foo:bar,,baz:bar");
  }

  @Test
  public void testStringDictionaryConverter_missingKeyOrValue() throws Exception {
    String badEntry = "foo";
    StringDictionaryConverter converter = new StringDictionaryConverter();
    expected.expect(OptionsParsingException.class);
    expected.expectMessage(String.format(
        "Dictionary entry [%s] does not contain both a key and a value.", badEntry));
    converter.convert(badEntry);
  }

  @Test
  public void testStringDictionaryConverter_extraFields() throws Exception {
    String badEntry = "foo:bar:baz";
    StringDictionaryConverter converter = new StringDictionaryConverter();
    expected.expect(OptionsParsingException.class);
    expected.expectMessage(String.format(
        "Dictionary entry [%s] contains too many fields.", badEntry));
    converter.convert(badEntry);
  }

  @Test
  public void testStringDictionaryConverter_duplicateKey() throws Exception {
    String key = "foo";
    String arg = String.format("%s:%s,%s:%s", key, "bar", key, "baz");
    StringDictionaryConverter converter = new StringDictionaryConverter();
    expected.expect(OptionsParsingException.class);
    expected.expectMessage(String.format(
        "Dictionary already contains the key [%s].", key));
    converter.convert(arg);
  }

  @Test
  public void testStringDictionaryConverter() throws Exception {
    StringDictionaryConverter converter = new StringDictionaryConverter();
    Map<String, String> result = converter.convert("foo:bar,baz:messy\\:stri\\,ng");
    assertThat(result).containsExactly("foo", "bar", "baz", "messy:stri,ng");
  }

  @Test
  public void testExistingPathStringDictionaryConverter() throws Exception {
    Path existingFile = tmp.newFile("existing").toPath();
    ExistingPathStringDictionaryConverter converter = new ExistingPathStringDictionaryConverter();
    Map<Path, String> result =
        converter
            // On Windows, the path starts like C:\foo\bar\..., we need to escape the colon.
            .convert(String.format("%s:string", existingFile.toString().replace(":", "\\:")));
    assertThat(result).containsExactly(existingFile, "string");
  }
}
