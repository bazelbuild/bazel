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

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Comparator.comparing;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager.Location;
import javax.tools.JavaFileObject;
import javax.tools.JavaFileObject.Kind;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;
import javax.tools.ToolProvider;

/**
 * Output a jar file containing all classes on the JDK 8 platform classpath of the default java
 * compiler of the current JDK.
 *
 * <p>usage: DumpPlatformClassPath <target release> output.jar
 */
public class DumpPlatformClassPath {

  public static void main(String[] args) throws IOException {
    if (args.length != 2) {
      System.err.println("usage: DumpPlatformClassPath <target release> <output jar>");
      System.exit(1);
    }
    String targetRelease = args[0];
    Map<String, byte[]> entries = new HashMap<>();
    JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
    StandardJavaFileManager fileManager = compiler.getStandardFileManager(null, null, UTF_8);
    if (isJdk9OrLater()) {
      // this configures the filemanager to use a JDK 8 bootclasspath
      compiler.getTask(
          null, fileManager, null, Arrays.asList("--release", targetRelease), null, null);
      for (Path path : getLocationAsPaths(fileManager)) {
        Files.walkFileTree(
            path,
            new SimpleFileVisitor<Path>() {
              @Override
              public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
                  throws IOException {
                if (file.getFileName().toString().endsWith(".sig")) {
                  String outputPath = path.relativize(file).toString();
                  outputPath =
                      outputPath.substring(0, outputPath.length() - ".sig".length()) + ".class";
                  entries.put(outputPath, Files.readAllBytes(file));
                }
                return FileVisitResult.CONTINUE;
              }
            });
      }
    } else {
      for (JavaFileObject fileObject :
          fileManager.list(
              StandardLocation.PLATFORM_CLASS_PATH,
              "",
              EnumSet.of(Kind.CLASS),
              /* recurse= */ true)) {
        String binaryName =
            fileManager.inferBinaryName(StandardLocation.PLATFORM_CLASS_PATH, fileObject);
        entries.put(
            binaryName.replace('.', '/') + ".class", toByteArray(fileObject.openInputStream()));
      }
    }
    try (OutputStream os = Files.newOutputStream(Paths.get(args[1]));
        BufferedOutputStream bos = new BufferedOutputStream(os, 65536);
        JarOutputStream jos = new JarOutputStream(bos)) {
      entries
          .entrySet()
          .stream()
          .sorted(comparing(Map.Entry::getKey))
          .forEachOrdered(e -> addEntry(jos, e.getKey(), e.getValue()));
    }
  }

  @SuppressWarnings("unchecked")
  private static Iterable<Path> getLocationAsPaths(StandardJavaFileManager fileManager) {
    try {
      return (Iterable<Path>)
          StandardJavaFileManager.class
              .getMethod("getLocationAsPaths", Location.class)
              .invoke(fileManager, StandardLocation.PLATFORM_CLASS_PATH);
    } catch (ReflectiveOperationException e) {
      throw new LinkageError(e.getMessage(), e);
    }
  }

  // Use a fixed timestamp for deterministic jar output.
  private static final long FIXED_TIMESTAMP =
      new GregorianCalendar(2010, 0, 1, 0, 0, 0).getTimeInMillis();

  private static void addEntry(JarOutputStream jos, String name, byte[] bytes) {
    try {
      JarEntry je = new JarEntry(name);
      je.setTime(FIXED_TIMESTAMP);
      je.setMethod(ZipEntry.STORED);
      je.setSize(bytes.length);
      CRC32 crc = new CRC32();
      crc.update(bytes);
      je.setCrc(crc.getValue());
      jos.putNextEntry(je);
      jos.write(bytes);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static byte[] toByteArray(InputStream is) throws IOException {
    byte[] buffer = new byte[8192];
    ByteArrayOutputStream boas = new ByteArrayOutputStream();
    while (true) {
      int r = is.read(buffer);
      if (r == -1) {
        break;
      }
      boas.write(buffer, 0, r);
    }
    return boas.toByteArray();
  }

  private static boolean isJdk9OrLater() {
    return Double.parseDouble(System.getProperty("java.class.version")) >= 53.0;
  }
}
