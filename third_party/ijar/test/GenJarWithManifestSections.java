// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.common.io.ByteStreams;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;
import java.util.jar.Attributes;
import java.util.jar.JarOutputStream;
import java.util.jar.Manifest;
import java.util.zip.ZipEntry;

/** Writes a jar file with manifest and optionally a Target-Label attribute. */
public final class GenJarWithManifestSections {

  public static void main(String[] args) throws IOException {
    try (JarOutputStream jos = new JarOutputStream(Files.newOutputStream(Paths.get(args[0])))) {
      addEntry(jos, "META-INF/MANIFEST.MF");
      Manifest manifest = new Manifest();
      Attributes mainAttributes = manifest.getMainAttributes();
      mainAttributes.put(Attributes.Name.MANIFEST_VERSION, "1.0");

      if (args.length > 1) {
        String targetLabel = args[1];
        mainAttributes.put(new Attributes.Name("Target-Label"), targetLabel);
      }

      Map<String, Attributes> entries = manifest.getEntries();

      Attributes foo = new Attributes();
      foo.put(new Attributes.Name("Foo"), "bar");
      entries.put("foo", foo);

      Attributes baz = new Attributes();
      baz.put(new Attributes.Name("Another"), "bar");
      entries.put("baz", baz);

      manifest.write(jos);

      addEntry(jos, "java/lang/String.class");
      ByteStreams.copy(String.class.getResourceAsStream("/java/lang/String.class"), jos);
    }
  }

  private static void addEntry(JarOutputStream jos, String name) throws IOException {
    ZipEntry ze = new ZipEntry(name);
    ze.setTime(0);
    jos.putNextEntry(ze);
  }

  private GenJarWithManifestSections() {}
}
