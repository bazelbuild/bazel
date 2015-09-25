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

package com.google.devtools.build.singlejar;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.devtools.build.singlejar.OptionFileExpander.OptionFileProvider;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Unit tests for {@link OptionFileExpander}.
 */
@RunWith(JUnit4.class)
public class OptionFileExpanderTest {

  private static class StoredOptionFileProvider implements OptionFileProvider {

    private Map<String, byte[]> availableFiles = new HashMap<>();

    void addFile(String filename, String content) {
      availableFiles.put(filename, content.getBytes(UTF_8));
    }

    @Override
    public InputStream getInputStream(String filename) throws IOException {
      byte[] result = availableFiles.get(filename);
      if (result == null) {
        throw new FileNotFoundException();
      }
      return new ByteArrayInputStream(result);
    }
  }

  @Test
  public void testNoExpansion() throws IOException {
    OptionFileExpander expander = new OptionFileExpander(new StoredOptionFileProvider());
    assertEquals(Arrays.asList("--some", "option", "list"),
        expander.expandArguments(Arrays.asList("--some", "option", "list")));
  }

  @Test
  public void testExpandSimpleOptionsFile() throws IOException {
    StoredOptionFileProvider provider = new StoredOptionFileProvider();
    provider.addFile("options", "--some option list");
    OptionFileExpander expander = new OptionFileExpander(provider);
    assertEquals(Arrays.asList("--some", "option", "list"),
        expander.expandArguments(Arrays.asList("@options")));
  }

  @Test
  public void testIllegalOptionsFile() {
    StoredOptionFileProvider provider = new StoredOptionFileProvider();
    provider.addFile("options", "'missing apostrophe");
    OptionFileExpander expander = new OptionFileExpander(provider);
    try {
      expander.expandArguments(Arrays.asList("@options"));
      fail();
    } catch (IOException e) {
      // Expected exception.
    }
  }
}
