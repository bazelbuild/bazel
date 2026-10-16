// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.coverage;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.Files.newBufferedWriter;
import static java.nio.file.StandardOpenOption.APPEND;
import static java.nio.file.StandardOpenOption.CREATE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.io.Files;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.JarInputStream;
import java.util.jar.Manifest;
import org.jacoco.agent.rt.IAgent;
import org.jacoco.agent.rt.RT;
import org.jacoco.core.tools.ExecFileLoader;
import sun.misc.Unsafe;

/**
 * Runner class used to generate code coverage report when using Jacoco offline instrumentation.
 *
 * <p>The complete list of features available for Jacoco offline instrumentation:
 * http://www.eclemma.org/jacoco/trunk/doc/offline.html
 *
 * <p>The structure is roughly following the canonical Jacoco example:
 * http://www.eclemma.org/jacoco/trunk/doc/examples/java/ReportGenerator.java
 *
 * <p>The following environment variables are expected:
 *
 * <ul>
 *   <li>JAVA_COVERAGE_FILE - specifies final location of the generated lcov file.
 * </ul>
 */
public class JacocoCoverageRunner {

  private final InputStream executionData;
  private final File reportFile;
  private ExecFileLoader execFileLoader;
  private ImmutableMap<String, byte[]> uninstrumentedClasses;
  private ImmutableSet<String> pathsForCoverage = ImmutableSet.of();

  public JacocoCoverageRunner(
      InputStream jacocoExec,
      String reportPath,
      Map<String, byte[]> uninstrumentedClasses,
      Set<String> pathsForCoverage) {
    executionData = jacocoExec;
    reportFile = new File(reportPath);
    this.uninstrumentedClasses = ImmutableMap.copyOf(uninstrumentedClasses);
    this.pathsForCoverage = ImmutableSet.copyOf(pathsForCoverage);
  }

  public void create() throws IOException {
    // Read the jacoco.exec file. Multiple data files could be merged at this point
    execFileLoader = new ExecFileLoader();
    execFileLoader.load(executionData);

    final Map<String, CoverageData> coverageData = analyze();
    createReport(coverageData);
  }

  /**
   * Returns a list of files from the given file list.
   *
   * @param fileList the file listing containing the paths to the jars
   * @param javaRunfilesRoot the root path to the Java runfiles
   * @throws IOException if an error occurs while reading the file list
   */
  public static ImmutableList<File> getFilesFromFileList(File fileList, String javaRunfilesRoot)
      throws IOException {
    List<String> files = Files.readLines(fileList, UTF_8);
    ImmutableList.Builder<File> unwrappedFiles = new ImmutableList.Builder<>();
    for (String file : files) {
      unwrappedFiles.add(new File(javaRunfilesRoot + "/" + file));
    }
    return unwrappedFiles.build();
  }

  /**
   * Returns the contents of -paths-for-coverage.txt files from the given jar.
   *
   * @param jar the jar containing the uninstrumented classes
   * @throws IOException if an error occurs while reading the jar
   */
  public static ImmutableSet<String> getPathsForCoverage(JarFile jar) throws IOException {
    ImmutableSet.Builder<String> pathsForCoverage = ImmutableSet.builder();
    Enumeration<JarEntry> jarFileEntries = jar.entries();
    while (jarFileEntries.hasMoreElements()) {
      JarEntry jarEntry = jarFileEntries.nextElement();
      String jarEntryName = jarEntry.getName();
      if (jarEntryName.endsWith("-paths-for-coverage.txt")) {
        BufferedReader bufferedReader =
            new BufferedReader(new InputStreamReader(jar.getInputStream(jarEntry), UTF_8));
        String line;
        while ((line = bufferedReader.readLine()) != null) {
          pathsForCoverage.add(line);
        }
      }
    }
    return pathsForCoverage.build();
  }

  /**
   * Returns a map of uninstrumented classes to their byte arrays from the given jar.
   *
   * @param jar the jar containing the uninstrumented classes
   * @throws IOException if an error occurs while reading the jar
   */
  public static ImmutableMap<String, byte[]> getUninstrumentedClasses(JarFile jar)
      throws IOException {
    ImmutableMap.Builder<String, byte[]> uninstrumentedClasses = ImmutableMap.builder();
    Enumeration<JarEntry> jarFileEntries = jar.entries();
    while (jarFileEntries.hasMoreElements()) {
      JarEntry jarEntry = jarFileEntries.nextElement();
      String jarEntryName = jarEntry.getName();
      if (jarEntryName.endsWith(".class.uninstrumented")) {
        if (jarEntryName.startsWith("META-INF/versions/")) {
          // TODO(cmita): remove if JaCoCo support for MR-JARs is fixed, see
          // https://github.com/jacoco/jacoco/issues/407
          continue;
        }
        uninstrumentedClasses.put(
            jarEntryName, ByteStreams.toByteArray(jar.getInputStream(jarEntry)));
      }
    }
    return uninstrumentedClasses.buildOrThrow();
  }

  @VisibleForTesting
  void createReport(final Map<String, CoverageData> coverageData) throws IOException {
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(pathsForCoverage);
    try (PrintWriter writer =
        new PrintWriter(newBufferedWriter(reportFile.toPath(), UTF_8, CREATE, APPEND))) {
      formatter.writeCoverageData(writer, coverageData);
    }
  }

  private Map<String, CoverageData> analyze() throws IOException {
    final CoverageAnalyzer analyzer = new CoverageAnalyzer(execFileLoader.getExecutionDataStore());

    Map<String, CoverageData> result = new TreeMap<>();
    for (Map.Entry<String, byte[]> entry : uninstrumentedClasses.entrySet()) {
      analyzer.analyzeClass(entry.getValue(), entry.getKey());
    }
    result.putAll(analyzer.getCoverage());

    return result;
  }

  private static Class<?> getMainClassFromClassLoader() throws IOException {
    // Note ClassLoader#getResource() will only return the first result, most likely a manifest
    // from the bootclasspath.
    if (JacocoCoverageRunner.class.getClassLoader() != null) {
      Enumeration<URL> manifests =
          JacocoCoverageRunner.class.getClassLoader().getResources("META-INF/MANIFEST.MF");
      while (manifests.hasMoreElements()) {
        Manifest manifest = new Manifest(manifests.nextElement().openStream());
        Attributes attributes = manifest.getMainAttributes();
        String className = attributes.getValue("Coverage-Main-Class");
        if (className != null) {
          // Some test frameworks use dummy Coverage-Main-Class in the deploy jars
          // which should be ignored by JacocoCoverageRunner.
          try {
            return Class.forName(className);
          } catch (ClassNotFoundException e) {
            // ignore this class and move on
          }
        }
      }
    }
    return null;
  }

  private static Class<?> getMainClass(boolean multipleJars) throws Exception {
    // There are several cases to consider:
    //
    // 1. This is the test executable for a java_test (or similar). Then there will be multiple
    // jars, including potentially deploy jars from runtime_deps. In this case, JACOCO_MAIN_CLASS
    // will be set and we should use it.
    //
    // 2. There is exactly one jar. This may be a runtime data dependency and we should prioritize
    // the main class provided by that jar.
    //
    // 3. There may be several jars, but JACOCO_MAIN_CLASS is not set. The fallback is to then
    // return main class specified in the first manifest. This can occur if jars are bundled in
    // a non java binary

    String jacocoMainClass = System.getenv("JACOCO_MAIN_CLASS");
    boolean jacocoMainClassSpecified = jacocoMainClass != null && !jacocoMainClass.isEmpty();
    Class<?> mainClass = null;

    if (multipleJars && jacocoMainClassSpecified) {
      mainClass = Class.forName(jacocoMainClass);
    } else {
      mainClass = getMainClassFromClassLoader();
      if (mainClass == null && jacocoMainClassSpecified) {
        // If we couldn't find a main class then try the one set in JACOCO_MAIN_CLASS
        mainClass = Class.forName(jacocoMainClass);
      }
    }
    if (mainClass == null) {
      throw new IllegalStateException(
          "JACOCO_METADATA_JAR/JACOCO_MAIN_CLASS environment variables not set, and no"
              + " META-INF/MANIFEST.MF on the classpath has a Coverage-Main-Class attribute. "
              + " Cannot determine the name of the main class for the code under test.");
    }
    return mainClass;
  }

  private static String getUniquePath(String pathTemplate, String suffix) throws IOException {
    // If pathTemplate is null, we're likely executing from a deploy jar and the test framework
    // did not properly set the environment for coverage reporting. This alone is not a reason for
    // throwing an exception, we're going to run anyway and write the coverage data to a temporary,
    // throw-away file.
    if (pathTemplate == null) {
      return File.createTempFile("coverage", suffix).getPath();
    } else {
      // bazel sets the path template to a file with the .dat extension. lcov_merger matches all
      // files having '.dat' in their name, so instead of appending we change the extension.
      File absolutePathTemplate = new File(pathTemplate).getAbsoluteFile();
      String prefix = absolutePathTemplate.getName();
      int lastDot = prefix.lastIndexOf('.');
      if (lastDot != -1) {
        prefix = prefix.substring(0, lastDot);
      }
      return File.createTempFile(prefix, suffix, absolutePathTemplate.getParentFile()).getPath();
    }
  }

  private static URL[] getUrls(ClassLoader classLoader, boolean jarIsWrapped, String wrappedJar) {
    // jarIsWrapped is a legacy parameter; it should be removed once we are sure Bazel will no
    // longer set JACOCO_IS_JAR_WRAPPED in java_stub_template
    URL[] urls = getClassLoaderUrls(classLoader);
    if (urls == null || urls.length == 0) {
      return urls;
    }
    // If the classpath was too long then a temporary top-level jar is created containing nothing
    // but a manifest with the original classpath. Those are the URLs we are looking for.
    URL classPathUrl = null;
    if (!Strings.isNullOrEmpty(wrappedJar)) {
      for (URL url : urls) {
        if (url.getPath().endsWith(wrappedJar)) {
          classPathUrl = url;
        }
      }
      if (classPathUrl == null) {
        System.err.println("Classpath JAR " + wrappedJar + " not provided");
        return null;
      }
    } else if (jarIsWrapped && urls.length == 1) {
      classPathUrl = urls[0];
    }
    if (classPathUrl != null) {
      try {
        String jarClassPath =
            new JarInputStream(classPathUrl.openStream())
                .getManifest()
                .getMainAttributes()
                .getValue("Class-Path");
        String[] urlStrings = jarClassPath.split(" ");
        URL[] newUrls = new URL[urlStrings.length];
        for (int i = 0; i < urlStrings.length; i++) {
          newUrls[i] = new URL(urlStrings[i]);
        }
        return newUrls;
      } catch (Exception e) {
        e.printStackTrace();
        return null;
      }
    }
    return urls;
  }

  private static URL[] getClassLoaderUrls(ClassLoader classLoader) {
    if (classLoader instanceof URLClassLoader) {
      return ((URLClassLoader) classLoader).getURLs();
    }

    // java 9 and later
    if (classLoader.getClass().getName().startsWith("jdk.internal.loader.ClassLoaders$")) {
      try {
        Field field = Unsafe.class.getDeclaredField("theUnsafe");
        field.setAccessible(true);
        Unsafe unsafe = (Unsafe) field.get(null);

        Field ucpField;
        try {
          // Java 9-15:
          // jdk.internal.loader.ClassLoaders.AppClassLoader.ucp
          ucpField = classLoader.getClass().getDeclaredField("ucp");
        } catch (NoSuchFieldException e) {
          // Java 16+:
          // jdk.internal.loader.BuiltinClassLoader.ucp
          // https://github.com/openjdk/jdk/commit/03a4df0acd103702e52dcd01c3f03fda4d7b04f5#diff-32cc12c0e3172fe5f2da1f65a75fa1cb920c39040d06323c83ad2c4d84e095aaL147
          ucpField = classLoader.getClass().getSuperclass().getDeclaredField("ucp");
        }
        long ucpFieldOffset = unsafe.objectFieldOffset(ucpField);
        Object ucpObject = unsafe.getObject(classLoader, ucpFieldOffset);

        Field pathField;
        try {
          // Java 9-26:
          // jdk.internal.loader.URLClassPath.path
          pathField = ucpField.getType().getDeclaredField("path");
        } catch (NoSuchFieldException e) {
          // Java 27+:
          // jdk.internal.loader.URLClassPath.searchPath
          pathField = ucpField.getType().getDeclaredField("searchPath");
        }
        long pathFieldOffset = unsafe.objectFieldOffset(pathField);
        ArrayList<URL> path = (ArrayList<URL>) unsafe.getObject(ucpObject, pathFieldOffset);

        return path.toArray(new URL[path.size()]);
      } catch (Exception e) {
        return null;
      }
    }
    return null;
  }

  public static void main(String[] args) throws Exception {
    String jarWrappedValue = System.getenv("JACOCO_IS_JAR_WRAPPED");
    String wrappedJarValue = System.getenv("CLASSPATH_JAR");
    boolean wasWrappedJar = jarWrappedValue != null ? !jarWrappedValue.equals("0") : false;

    final HashMap<String, byte[]> uninstrumentedClasses = new HashMap<>();
    ImmutableSet.Builder<String> pathsForCoverageBuilder = ImmutableSet.builder();
    ClassLoader classLoader = ClassLoader.getSystemClassLoader();
    URL[] urls = getUrls(classLoader, wasWrappedJar, wrappedJarValue);
    if (urls != null) {
      for (URL url : urls) {
        try (JarFile jar = new JarFile(new File(url.toURI().getPath()))) {
          uninstrumentedClasses.putAll(getUninstrumentedClasses(jar));
          pathsForCoverageBuilder.addAll(getPathsForCoverage(jar));
        }
      }
    }

    final ImmutableSet<String> pathsForCoverage = pathsForCoverageBuilder.build();

    final String coverageReportBase = System.getenv("JAVA_COVERAGE_FILE");

    // Disable Jacoco's default output mechanism, which runs as a shutdown hook. We generate the
    // report in our own shutdown hook below, and we want to avoid the data race (shutdown hooks are
    // not guaranteed any particular order). Note that also by default, Jacoco appends coverage
    // data, which can have surprising results if running tests locally or somehow encountering
    // the previous .exec file.
    System.setProperty("jacoco-agent.output", "none");

    // We have no use for this sessionId property, but leaving it blank results in a DNS lookup
    // at runtime. A minor annoyance: the documentation insists the property name is "sessionId",
    // however on closer inspection of the source code, it turns out to be "sessionid"...
    System.setProperty("jacoco-agent.sessionid", "default");

    // A JVM shutdown hook has a fixed amount of time (OS-dependent) before it is terminated.
    // For our purpose, it's more than enough to scan through the instrumented jar and match up
    // the bytecode with the coverage data. It wouldn't be enough for scanning the entire classpath,
    // or doing something else terribly inefficient.
    Runtime.getRuntime()
        .addShutdownHook(
            new Thread() {
              @Override
              public void run() {
                try {
                  // If the test spawns multiple JVMs, they will race to write to the same files. We
                  // need to generate unique paths for each execution. lcov_merger simply collects
                  // all the .dat files in the current directory anyway, so we don't need to worry
                  // about merging them.
                  String coverageReport = getUniquePath(coverageReportBase, ".dat");
                  String coverageData = getUniquePath(coverageReportBase, ".exec");

                  // Get a handle on the Jacoco Agent and write out the coverage data. Other options
                  // included talking to the agent via TCP (useful when gathering coverage from
                  // multiple JVMs), or via JMX (the agent's MXBean is called
                  // 'org.jacoco:type=Runtime'). As we're running in the same JVM, these options
                  // seemed overkill, we can just refer to the Jacoco runtime as RT.
                  // See http://www.eclemma.org/jacoco/trunk/doc/agent.html for all the options
                  // available.
                  ByteArrayInputStream dataInputStream;
                  try {
                    IAgent agent = RT.getAgent();
                    byte[] data = agent.getExecutionData(false);
                    try (FileOutputStream fs = new FileOutputStream(coverageData, true)) {
                      fs.write(data);
                    }
                    // We append to the output file, but run report generation only for the coverage
                    // data from this JVM. The output file may contain data from other
                    // subprocesses, etc.
                    dataInputStream = new ByteArrayInputStream(data);
                  } catch (IllegalStateException e) {
                    // In this case, we didn't execute a single instrumented file, so the agent
                    // isn't live. There's no coverage to report, but it's otherwise a successful
                    // invocation.
                    dataInputStream = new ByteArrayInputStream(new byte[0]);
                  }
                  JacocoCoverageRunner jacocoCoverageRunner =
                      new JacocoCoverageRunner(
                          dataInputStream, coverageReport, uninstrumentedClasses, pathsForCoverage);
                  jacocoCoverageRunner.create();
                } catch (IOException e) {
                  e.printStackTrace();
                  Runtime.getRuntime().halt(1);
                }
              }
            });

    Class<?> mainClass = getMainClass(urls != null && urls.length > 1);
    Method main = mainClass.getMethod("main", String[].class);
    main.setAccessible(true);
    // Another option would be to run the tests in a separate JVM, let Jacoco dump out the coverage
    // data, wait for the subprocess to finish and then generate the lcov report. The only benefit
    // of doing this is not being constrained by the hard 5s limit of the shutdown hook. Setting up
    // the subprocess to match all JVM flags, runtime classpath, bootclasspath, etc is doable.
    // We'd share the same limitation if the system under test uses shutdown hooks internally, as
    // there's no way to collect coverage data on that code.
    main.invoke(null, new Object[] {args});
  }
}
