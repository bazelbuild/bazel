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
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
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
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
 * <p>The uninstrumented classes to be analyzed and the paths of the covered source files are
 * collected from the jars on the runtime classpath.
 *
 * <p>The following environment variables are expected:
 *
 * <ul>
 *   <li>JAVA_COVERAGE_FILE - specifies final location of the generated lcov file.
 *   <li>JACOCO_MAIN_CLASS - specifies the main class of the code under test if not running from a
 *       deploy jar with a Coverage-Main-Class manifest attribute.
 *   <li>CLASSPATH_JAR - specifies the name of the jar on the classpath whose manifest contains the
 *       actual runtime classpath, if the classpath was too long to be passed directly.
 * </ul>
 */
public class JacocoCoverageRunner {

  private final InputStream executionData;
  private final File reportFile;
  private final Map<String, byte[]> uninstrumentedClasses;
  private final ImmutableSet<String> pathsForCoverage;
  private ExecFileLoader execFileLoader;

  public JacocoCoverageRunner(
      InputStream jacocoExec,
      String reportPath,
      Map<String, byte[]> uninstrumentedClasses,
      ImmutableSet<String> pathsForCoverage) {
    executionData = jacocoExec;
    reportFile = new File(reportPath);
    this.uninstrumentedClasses = uninstrumentedClasses;
    this.pathsForCoverage = pathsForCoverage;
  }

  public void create() throws IOException {
    // Read the jacoco.exec file. Multiple data files could be merged at this point
    execFileLoader = new ExecFileLoader();
    execFileLoader.load(executionData);

    final Map<String, CoverageData> coverageData = analyze();
    createReport(coverageData);
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
    for (Map.Entry<String, byte[]> entry : uninstrumentedClasses.entrySet()) {
      analyzer.analyzeClass(entry.getValue(), entry.getKey());
    }
    return new TreeMap<>(analyzer.getCoverage());
  }

  /**
   * Collects the uninstrumented class files and the paths of the covered source files from the
   * given jar.
   *
   * <p>The uninstrumented classes are named using the .class.uninstrumented suffix.
   *
   * <p>If a jar contains uninstrumented classes it will also contain a txt file with the paths of
   * each of these classes, called "-paths-for-coverage.txt". This file expects one path per line
   * specified as either:
   *
   * <ul>
   *   <li>A single path (e.g. /dir/com/example/Foo.java).
   *   <li>A mapping between source and class paths delimited with by /// (e.g.
   *       /dir/Foo.java////com/example/Foo.java).
   * </ul>
   */
  private static void collectCoverageMetadata(
      File jar,
      Map<String, byte[]> uninstrumentedClasses,
      ImmutableSet.Builder<String> pathsForCoverage)
      throws IOException {
    try (JarFile jarFile = new JarFile(jar)) {
      Enumeration<JarEntry> jarFileEntries = jarFile.entries();
      while (jarFileEntries.hasMoreElements()) {
        JarEntry jarEntry = jarFileEntries.nextElement();
        String jarEntryName = jarEntry.getName();
        if (jarEntryName.endsWith(".class.uninstrumented")
            && !uninstrumentedClasses.containsKey(jarEntryName)) {
          try (InputStream in = jarFile.getInputStream(jarEntry)) {
            uninstrumentedClasses.put(jarEntryName, ByteStreams.toByteArray(in));
          }
        } else if (jarEntryName.endsWith("-paths-for-coverage.txt")) {
          try (BufferedReader bufferedReader =
              new BufferedReader(new InputStreamReader(jarFile.getInputStream(jarEntry), UTF_8))) {
            String line;
            while ((line = bufferedReader.readLine()) != null) {
              pathsForCoverage.add(line);
            }
          }
        }
      }
    }
  }

  private static Class<?> getMainClass(boolean insideDeployJar) throws Exception {
    Class<?> mainClass;
    // If we're running inside a deploy jar we have to open the manifest and read the value of
    // "Coverage-Main-Class", set by bazel.
    // Note ClassLoader#getResource() will only return the first result, most likely a manifest
    // from the bootclasspath.
    if (insideDeployJar) {
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
              mainClass = Class.forName(className);
              return mainClass;
            } catch (ClassNotFoundException e) {
              // ignore this class and move on
            }
          }
        }
      }
    }
    // Check JACOCO_MAIN_CLASS after making sure we're not running inside a deploy jar, otherwise
    // the deploy jar will be invoked using the wrong main class.
    String jacocoMainClass = System.getenv("JACOCO_MAIN_CLASS");
    if (jacocoMainClass != null) {
      return Class.forName(jacocoMainClass);
    }
    throw new IllegalStateException(
        "JACOCO_MAIN_CLASS environment variable not set, and no META-INF/MANIFEST.MF on the"
            + " classpath has a Coverage-Main-Class attribute. Cannot determine the name of the"
            + " main class for the code under test.");
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

  private static URL[] getUrls(ClassLoader classLoader, String classpathJar)
      throws MalformedURLException {
    URL[] urls = getClassLoaderUrls(classLoader);
    if (urls == null) {
      // The search path of the system class loader is initialized from java.class.path. Fall back
      // to it if the class loader can't be inspected, e.g. because a custom system class loader is
      // used or the memory access methods of sun.misc.Unsafe are unavailable.
      urls = getJavaClassPathUrls();
    }
    if (urls.length == 0 || Strings.isNullOrEmpty(classpathJar)) {
      return urls;
    }
    // If the classpath was too long then a temporary top-level jar is created containing nothing
    // but a manifest with the original classpath. Those are the URLs we are looking for.
    URL classPathUrl = null;
    for (URL url : urls) {
      if (url.getPath().endsWith(classpathJar)) {
        classPathUrl = url;
      }
    }
    if (classPathUrl == null) {
      System.err.println("Classpath JAR " + classpathJar + " not provided");
      return null;
    }
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

  private static URL[] getJavaClassPathUrls() throws MalformedURLException {
    List<String> entries =
        Splitter.on(File.pathSeparatorChar)
            .omitEmptyStrings()
            .splitToList(System.getProperty("java.class.path", ""));
    URL[] urls = new URL[entries.size()];
    for (int i = 0; i < urls.length; i++) {
      urls[i] = new File(entries.get(i)).toURI().toURL();
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

        // jdk.internal.loader.URLClassPath.path
        Field pathField = ucpField.getType().getDeclaredField("path");
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
    URL[] urls = getUrls(ClassLoader.getSystemClassLoader(), System.getenv("CLASSPATH_JAR"));
    if (urls == null) {
      throw new IllegalStateException(
          "Failed to determine the runtime classpath. Cannot collect coverage for the code under"
              + " test.");
    }

    // Collect
    // - uninstrumented class files for coverage before starting the actual test
    // - paths considered for coverage
    // Collecting these in the shutdown hook is too expensive (we only have a 5s budget).
    int deployJars = 0;
    final HashMap<String, byte[]> uninstrumentedClasses = new HashMap<>();
    ImmutableSet.Builder<String> pathsForCoverageBuilder = ImmutableSet.builder();
    for (URL url : urls) {
      String file = url.toURI().getPath();
      if (file.endsWith("_deploy.jar")) {
        deployJars++;
      }
      if (file.endsWith(".jar")) {
        collectCoverageMetadata(new File(file), uninstrumentedClasses, pathsForCoverageBuilder);
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
    // For our purpose, it's more than enough to match up the uninstrumented bytecode collected
    // above with the coverage data. It wouldn't be enough for scanning the entire classpath, or
    // doing something else terribly inefficient.
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

                  new JacocoCoverageRunner(
                          dataInputStream, coverageReport, uninstrumentedClasses, pathsForCoverage)
                      .create();
                } catch (IOException e) {
                  e.printStackTrace();
                  Runtime.getRuntime().halt(1);
                }
              }
            });

    // If running inside a deploy jar the classpath contains only that deploy jar.
    // It can happen that multiple deploy jars are on the classpath. In that case we are running
    // from a regular java binary where all the environment (e.g. JACOCO_MAIN_CLASS) is set
    // accordingly.
    boolean insideDeployJar = deployJars == 1 && urls.length == 1;
    Class<?> mainClass = getMainClass(insideDeployJar);
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
