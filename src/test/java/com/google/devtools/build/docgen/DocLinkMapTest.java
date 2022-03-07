// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test class for DocLinkMap. */
@RunWith(JUnit4.class)
public class DocLinkMapTest {

  @Rule public final TemporaryFolder tmp = new TemporaryFolder();

  @Test
  public void testCreateFromFile() throws IOException {
    String content =
        "{ \"beRoot\": \"/be_root\", \"values\": { \"build-ref\": \"build-ref.html\" } }";

    Path path = tmp.newFile("map.json").toPath();
    Files.write(path, content.getBytes(UTF_8));

    DocLinkMap map = DocLinkMap.createFromFile(path.toString());
    assertThat(map.beRoot).isEqualTo("/be_root");
    assertThat(map.values).containsExactly("build-ref", "build-ref.html");
  }
}
