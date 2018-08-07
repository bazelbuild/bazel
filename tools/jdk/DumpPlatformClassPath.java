// Copyright 2017 The Bazel Authors. All rights reserved.
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

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.nio.file.DirectoryStream;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.GregorianCalendar;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;

/**
 * Output a jar file containing all classes on the platform classpath of the current JDK.
 *
 * <p>usage: DumpPlatformClassPath <output jar>
 */
public class DumpPlatformClassPath {

  public static void main(String[] args) throws IOException {
    if (args.length != 1) {
      System.err.println("usage: DumpPlatformClassPath <output jar>");
      System.exit(1);
    }
    Path output = Paths.get(args[0]);
    Path path = Paths.get(System.getProperty("java.home"));
    if (path.endsWith("jre")) {
      path = path.getParent();
    }
    Path rtJar = path.resolve("jre/lib/rt.jar");
    System.err.println(rtJar);
    if (Files.exists(rtJar)) {
      Files.copy(rtJar, output);
      return;
    }
    Path modules = FileSystems.getFileSystem(URI.create("jrt:/")).getPath("modules");
    try (OutputStream os = Files.newOutputStream(output);
        BufferedOutputStream bos = new BufferedOutputStream(os, 65536);
        JarOutputStream jos = new JarOutputStream(bos)) {
      try (DirectoryStream<Path> modulePaths = Files.newDirectoryStream(modules)) {
        for (Path modulePath : modulePaths) {
          Files.walkFileTree(
              modulePath,
              new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
                    throws IOException {
                  String name = path.getFileName().toString();
                  if (name.endsWith(".class") && !name.equals("module-info.class")) {
                    addEntry(jos, modulePath.relativize(path).toString(), Files.readAllBytes(path));
                  }
                  return super.visitFile(path, attrs);
                }
              });
        }
      }
    }
  }

  // Use a fixed timestamp for deterministic jar output.
  private static final long FIXED_TIMESTAMP =
      new GregorianCalendar(2010, 0, 1, 0, 0, 0).getTimeInMillis();

  private static void addEntry(JarOutputStream jos, String name, byte[] bytes) throws IOException {
    JarEntry je = new JarEntry(name);
    je.setTime(FIXED_TIMESTAMP);
    je.setMethod(ZipEntry.STORED);
    je.setSize(bytes.length);
    CRC32 crc = new CRC32();
    crc.update(bytes);
    je.setCrc(crc.getValue());
    jos.putNextEntry(je);
    jos.write(bytes);
  }
}
