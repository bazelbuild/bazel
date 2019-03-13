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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.io.Files;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
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
import org.jacoco.core.analysis.Analyzer;
import org.jacoco.core.analysis.CoverageBuilder;
import org.jacoco.core.analysis.IBundleCoverage;
import org.jacoco.core.tools.ExecFileLoader;
import org.jacoco.report.IReportVisitor;
import org.jacoco.report.ISourceFileLocator;

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
 * <ul>
 * <li>JAVA_COVERAGE_FILE - specifies final location of the generated lcov file.</li>
 * <li>JACOCO_METADATA_JAR - specifies jar containing uninstrumented classes to be analyzed.</li>
 * </ul>
 */
public class JacocoCoverageRunner {

  private final ImmutableList<File> classesJars;
  private final InputStream executionData;
  private final File reportFile;
  private final boolean isNewCoverageImplementation;
  private ExecFileLoader execFileLoader;
  private HashMap<String, byte[]> uninstrumentedClasses;
  private ImmutableSet<String> pathsForCoverage = ImmutableSet.of();

  public JacocoCoverageRunner(InputStream jacocoExec, String reportPath, File... metadataJars) {
    this(false, jacocoExec, reportPath, metadataJars);
  }

  /**
   * Creates a new coverage runner extracting the classes jars from a wrapper file. Uses
   * javaRunfilesRoot to compute the absolute path of the jars inside the wrapper file.
   */
  public JacocoCoverageRunner(
      boolean isNewCoverageImplementation,
      InputStream jacocoExec,
      String reportPath,
      File wrapperFile,
      String javaRunfilesRoot)
      throws IOException {
    executionData = jacocoExec;
    reportFile = new File(reportPath);
    this.isNewCoverageImplementation = isNewCoverageImplementation;
    this.classesJars = getFilesFromFileList(wrapperFile, javaRunfilesRoot);
  }

  public JacocoCoverageRunner(
      boolean isNewCoverageImplementation,
      InputStream jacocoExec,
      String reportPath,
      File... metadataJars) {
    executionData = jacocoExec;
    reportFile = new File(reportPath);
    this.isNewCoverageImplementation = isNewCoverageImplementation;
    this.classesJars = ImmutableList.copyOf(metadataJars);
  }

  public JacocoCoverageRunner(
      boolean isNewCoverageImplementation,
      InputStream jacocoExec,
      String reportPath,
      HashMap<String, byte[]> uninstrumentedClasses,
      ImmutableSet<String> pathsForCoverage,
      File... metadataJars) {
    executionData = jacocoExec;
    reportFile = new File(reportPath);
    this.isNewCoverageImplementation = isNewCoverageImplementation;
    this.classesJars = ImmutableList.copyOf(metadataJars);
    this.uninstrumentedClasses = uninstrumentedClasses;
    this.pathsForCoverage = pathsForCoverage;
  }

  public void create() throws IOException {
    // Read the jacoco.exec file. Multiple data files could be merged at this point
    execFileLoader = new ExecFileLoader();
    execFileLoader.load(executionData);

    // Run the structure analyzer on a single class folder or jar file to build up the coverage
    // model. Typically you would create a bundle for each class folder and each jar you want in
    // your report. If you have more than one bundle you may need to add a grouping node to the
    // report. The lcov formatter doesn't seem to care, and we're only using one bundle anyway.
    final IBundleCoverage bundleCoverage = analyzeStructure();

    final Map<String, BranchCoverageDetail> branchDetails = analyzeBranch();
    createReport(bundleCoverage, branchDetails);
  }

  @VisibleForTesting
  void createReport(
      final IBundleCoverage bundleCoverage, final Map<String, BranchCoverageDetail> branchDetails)
      throws IOException {
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(createPathsSet());
    final IReportVisitor visitor = formatter.createVisitor(reportFile, branchDetails);

    // Initialize the report with all of the execution and session information. At this point the
    // report doesn't know about the structure of the report being created.
    visitor.visitInfo(
        execFileLoader.getSessionInfoStore().getInfos(),
        execFileLoader.getExecutionDataStore().getContents());

    // Populate the report structure with the bundle coverage information.
    // Call visitGroup if you need groups in your report.

    // Note the API requires a sourceFileLocator because the HTML and XML formatters display a page
    // of code annotated with coverage information. Having the source files is not actually needed
    // for generating the lcov report...
    visitor.visitBundle(
        bundleCoverage,
        new ISourceFileLocator() {

          @Override
          public Reader getSourceFile(String packageName, String fileName) throws IOException {
            return null;
          }

          @Override
          public int getTabWidth() {
            return 0;
          }
        });

    // Signal end of structure information to allow report to write all information out
    visitor.visitEnd();
  }

  @VisibleForTesting
  IBundleCoverage analyzeStructure() throws IOException {
    final CoverageBuilder coverageBuilder = new CoverageBuilder();
    final Analyzer analyzer = new Analyzer(execFileLoader.getExecutionDataStore(), coverageBuilder);
    Set<String> alreadyInstrumentedClasses = new HashSet<>();
    if (uninstrumentedClasses == null) {
      for (File classesJar : classesJars) {
        if (isNewCoverageImplementation) {
          analyzeUninstrumentedClassesFromJar(analyzer, classesJar, alreadyInstrumentedClasses);
        } else {
          analyzer.analyzeAll(classesJar);
        }
      }
    } else {
      for (Map.Entry<String, byte[]> entry : uninstrumentedClasses.entrySet()) {
        analyzer.analyzeClass(entry.getValue(), entry.getKey());
      }
    }

    // TODO(bazel-team): Find out where the name of the bundle can pop out in the report.
    return coverageBuilder.getBundle("isthisevenused");
  }

  // Additional pass to process the branch details of the classes
  private Map<String, BranchCoverageDetail> analyzeBranch() throws IOException {
    final BranchDetailAnalyzer analyzer =
        new BranchDetailAnalyzer(execFileLoader.getExecutionDataStore());

    Map<String, BranchCoverageDetail> result = new TreeMap<>();
    Set<String> alreadyInstrumentedClasses = new HashSet<>();
    if (uninstrumentedClasses == null) {
      for (File classesJar : classesJars) {
        if (isNewCoverageImplementation) {
          analyzeUninstrumentedClassesFromJar(analyzer, classesJar, alreadyInstrumentedClasses);
        } else {
          analyzer.analyzeAll(classesJar);
        }
        result.putAll(analyzer.getBranchDetails());
      }
    } else {
      for (Map.Entry<String, byte[]> entry : uninstrumentedClasses.entrySet()) {
        analyzer.analyzeClass(entry.getValue(), entry.getKey());
      }
      result.putAll(analyzer.getBranchDetails());
    }
    return result;
  }

  /**
   * Analyzes all uninstrumented class files found in the given jar.
   *
   * <p>The uninstrumented classes are named using the .class.uninstrumented suffix.
   */
  private void analyzeUninstrumentedClassesFromJar(
      Analyzer analyzer, File jar, Set<String> alreadyInstrumentedClasses) throws IOException {
    JarFile jarFile = new JarFile(jar);
    Enumeration<JarEntry> jarFileEntries = jarFile.entries();
    while (jarFileEntries.hasMoreElements()) {
      JarEntry jarEntry = jarFileEntries.nextElement();
      String jarEntryName = jarEntry.getName();
      if (jarEntryName.endsWith(".class.uninstrumented")
          && !alreadyInstrumentedClasses.contains(jarEntryName)) {
        analyzer.analyzeAll(jarFile.getInputStream(jarEntry), jarEntryName);
        alreadyInstrumentedClasses.add(jarEntryName);
      }
    }
  }

  /**
   * Creates a {@link Set} containing the paths of the covered Java files.
   *
   * <p>The paths are retrieved from a txt file that is found inside each jar containing
   * uninstrumented classes. Each line of the txt file represents a path to be added to the set.
   *
   * <p>This set is needed by {@link JacocoLCOVFormatter} in order to output the correct path for
   * each covered class.
   */
  @VisibleForTesting
  ImmutableSet<String> createPathsSet() throws IOException {
    if (!isNewCoverageImplementation) {
      return ImmutableSet.<String>of();
    }
    if (!pathsForCoverage.isEmpty()) {
      return pathsForCoverage;
    }
    ImmutableSet.Builder<String> execPathsSetBuilder = ImmutableSet.builder();
    for (File classJar : classesJars) {
      addEntriesToExecPathsSet(classJar, execPathsSetBuilder);
    }
    ImmutableSet<String> result = execPathsSetBuilder.build();
    return result;
  }

  /**
   * Adds to the given {@link Set} the paths found in a txt file inside the given jar.
   *
   * <p>If a jar contains uninstrumented classes it will also contain a txt file with the paths of
   * each of these classes, one on each line.
   */
  @VisibleForTesting
  static void addEntriesToExecPathsSet(
      File jar, ImmutableSet.Builder<String> execPathsSetBuilder) throws IOException {
    JarFile jarFile = new JarFile(jar);
    Enumeration<JarEntry> jarFileEntries = jarFile.entries();
    while (jarFileEntries.hasMoreElements()) {
      JarEntry jarEntry = jarFileEntries.nextElement();
      String jarEntryName = jarEntry.getName();
      if (jarEntryName.endsWith("-paths-for-coverage.txt")) {
        BufferedReader bufferedReader =
            new BufferedReader(new InputStreamReader(jarFile.getInputStream(jarEntry), UTF_8));
        String line;
        while ((line = bufferedReader.readLine()) != null) {
          execPathsSetBuilder.add(line);
        }
      }
    }
  }

  private static String getMainClass(String metadataJar, boolean isNewImplementation)
      throws Exception {
    final String jacocoMainClass = System.getenv("JACOCO_MAIN_CLASS");
    if (jacocoMainClass != null) {
      return jacocoMainClass;
    }
    if (!isNewImplementation && metadataJar != null) {
      // Blaze guarantees that JACOCO_METADATA_JAR has a proper manifest with a Main-Class entry.
      try (JarInputStream jarStream = new JarInputStream(new FileInputStream(metadataJar))) {
        return jarStream.getManifest().getMainAttributes().getValue("Main-Class");
      }
    } else {
      // If metadataJar was not set, we're running inside a deploy jar. We have to open the manifest
      // and read the value of "Precoverage-Class", set by Blaze. Note ClassLoader#getResource()
      // will only return the first result, most likely a manifest from the bootclasspath.
      Enumeration<URL> manifests =
          JacocoCoverageRunner.class.getClassLoader().getResources("META-INF/MANIFEST.MF");
      while (manifests.hasMoreElements()) {
        Manifest manifest = new Manifest(manifests.nextElement().openStream());
        Attributes attributes = manifest.getMainAttributes();
        String className = attributes.getValue("Coverage-Main-Class");
        if (className != null) {
          return className;
        }
      }
      throw new IllegalStateException(
          "JACOCO_METADATA_JAR/JACOCO_MAIN_CLASS environment variables not set, and no"
              + " META-INF/MANIFEST.MF on the classpath has a Coverage-Main-Class attribute. "
              + " Cannot determine the name of the main class for the code under test.");
    }
  }

  private static String getUniquePath(String pathTemplate, String suffix) throws IOException {
    // If pathTemplate is null, we're likely executing from a deploy jar and the test framework
    // did not properly set the environment for coverage reporting. This alone is not a reason for
    // throwing an exception, we're going to run anyway and write the coverage data to a temporary,
    // throw-away file.
    if (pathTemplate == null) {
      return File.createTempFile("coverage", suffix).getPath();
    } else {
      // Blaze sets the path template to a file with the .dat extension. lcov_merger matches all
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

  /**
   * Returns an immutable list containing all the file paths found in mainFile. It uses the
   * javaRunfilesRoot prefix for every found file to compute its absolute path.
   */
  private static ImmutableList<File> getFilesFromFileList(File mainFile, String javaRunfilesRoot)
      throws IOException {
    List<String> metadataFiles = Files.readLines(mainFile, UTF_8);
    ImmutableList.Builder<File> convertedMetadataFiles = new Builder<>();
    for (String metadataFile : metadataFiles) {
      convertedMetadataFiles.add(new File(javaRunfilesRoot + "/" + metadataFile));
    }
    return convertedMetadataFiles.build();
  }

  public static void main(String[] args) throws Exception {
    String metadataFile = System.getenv("JACOCO_METADATA_JAR");

    String javaCoverageNewImplementationValue = System.getenv("JAVA_COVERAGE_NEW_IMPLEMENTATION");
    final boolean isNewImplementation =
        (javaCoverageNewImplementationValue != null
                && javaCoverageNewImplementationValue.equals("YES"))
            || (metadataFile == null
                ? false
                : (metadataFile.endsWith(".txt") || metadataFile.endsWith("_merged_instr.jar")));

    File[] metadataFiles = null;
    final HashMap<String, byte[]> uninstrumentedClasses = new HashMap<>();
    ImmutableSet.Builder<String> pathsForCoverageBuilder = new ImmutableSet.Builder<>();
    if (isNewImplementation) {
      ClassLoader classLoader = ClassLoader.getSystemClassLoader();
      if (classLoader instanceof URLClassLoader) {
        URL[] urls = ((URLClassLoader) classLoader).getURLs();
        metadataFiles = new File[urls.length];
        for (int i = 0; i < urls.length; i++) {
          URL url = urls[i];
          metadataFiles[i] = new File(url.getFile());
          // Special case for deploy jars.
          if (url.getFile().endsWith("_deploy.jar")) {
            metadataFile = url.getFile();
          } else if (url.getFile().endsWith(".jar")) {
            // Collect
            // - uninstrumented class files for coverage before starting the actual test
            // - paths considered for coverage
            // Collecting these in the shutdown hook is too expensive (we only have a 5s budget).
            JarFile jarFile = new JarFile(url.getFile());
            Enumeration<JarEntry> jarFileEntries = jarFile.entries();
            while (jarFileEntries.hasMoreElements()) {
              JarEntry jarEntry = jarFileEntries.nextElement();
              String jarEntryName = jarEntry.getName();
              if (jarEntryName.endsWith(".class.uninstrumented")
                  && !uninstrumentedClasses.containsKey(jarEntryName)) {
                uninstrumentedClasses.put(
                    jarEntryName, ByteStreams.toByteArray(jarFile.getInputStream(jarEntry)));
              } else if (jarEntryName.endsWith("-paths-for-coverage.txt")) {
                BufferedReader bufferedReader =
                    new BufferedReader(
                        new InputStreamReader(jarFile.getInputStream(jarEntry), UTF_8));
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                  pathsForCoverageBuilder.add(line);
                }
              }
            }
          }
        }
      }
    }
    final ImmutableSet<String> pathsForCoverage = pathsForCoverageBuilder.build();
    final String metadataFileFinal = metadataFile;
    final File[] metadataFilesFinal = metadataFiles;
    final String javaRunfilesRoot = System.getenv("JACOCO_JAVA_RUNFILES_ROOT");

    boolean hasOneFile = false;
    if (!isNewImplementation) {
      // --noexperimental_java_coverage sets JACOCO_METADATA_JAR to the instrumented jar
      // and it is only one file.
      hasOneFile = true;
    } else if (metadataFile != null
        && (metadataFile.endsWith("_merged_instr.jar") || metadataFile.endsWith("_deploy.jar"))) {
      // --experimental_java_coverage can set JACOCO_METADATA_JAR to either one file (a deploy jar
      // or a merged jar) or to multiple jars.
      hasOneFile = true;
    }
    final boolean hasOneFileFinal = hasOneFile;

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

                  if (metadataFileFinal != null || metadataFilesFinal != null) {
                    File[] metadataJars;
                    if (metadataFilesFinal != null) {
                      metadataJars = metadataFilesFinal;
                    } else {
                      metadataJars =
                          hasOneFileFinal
                              ? new File[] {new File(metadataFileFinal)}
                              : getFilesFromFileList(new File(metadataFileFinal), javaRunfilesRoot)
                                  .toArray(new File[0]);
                    }

                    if (uninstrumentedClasses.isEmpty()) {
                      new JacocoCoverageRunner(
                              isNewImplementation, dataInputStream, coverageReport, metadataJars)
                          .create();
                    } else {
                      new JacocoCoverageRunner(
                              isNewImplementation,
                              dataInputStream,
                              coverageReport,
                              uninstrumentedClasses,
                              pathsForCoverage,
                              metadataJars)
                          .create();
                    }
                  }
                } catch (IOException e) {
                  e.printStackTrace();
                  Runtime.getRuntime().halt(1);
                }
              }
            });

    // Another option would be to run the tests in a separate JVM, let Jacoco dump out the coverage
    // data, wait for the subprocess to finish and then generate the lcov report. The only benefit
    // of doing this is not being constrained by the hard 5s limit of the shutdown hook. Setting up
    // the subprocess to match all JVM flags, runtime classpath, bootclasspath, etc is doable.
    // We'd share the same limitation if the system under test uses shutdown hooks internally, as
    // there's no way to collect coverage data on that code.
    String mainClass = getMainClass(metadataFile, isNewImplementation);
    Method main = Class.forName(mainClass).getMethod("main", String[].class);
    main.setAccessible(true);
    main.invoke(null, new Object[] {args});
  }
}
