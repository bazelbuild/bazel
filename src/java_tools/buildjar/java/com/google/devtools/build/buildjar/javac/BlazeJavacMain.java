// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.collect.MoreCollectors.toOptional;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Comparator.comparing;
import static java.util.Locale.ENGLISH;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.javac.BlazeJavacResult.Status;
import com.google.devtools.build.buildjar.javac.CancelCompilerPlugin.CancelRequestException;
import com.google.devtools.build.buildjar.javac.FormattedDiagnostic.Listener;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.buildjar.javac.statistics.BlazeJavacStatistics;
import com.sun.source.util.JavacTask;
import com.sun.tools.javac.api.ClientCodeWrapper.Trusted;
import com.sun.tools.javac.api.JavacTaskImpl;
import com.sun.tools.javac.api.JavacTool;
import com.sun.tools.javac.file.CacheFSInfo;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.main.Main.Result;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Log;
import com.sun.tools.javac.util.Options;
import com.sun.tools.javac.util.PropagatedException;
import java.io.IOError;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticListener;
import javax.tools.StandardLocation;

/**
 * Main class for our custom patched javac.
 *
 * <p>This main class tweaks the standard javac log class by changing the compiler's context to use
 * our custom log class. This custom log class modifies javac's output to list all errors after all
 * warnings.
 */
public class BlazeJavacMain {

  private static final Pattern INCOMPATIBLE_SYSTEM_CLASS_PATH_ERROR =
      Pattern.compile(
          "(?s)bad class file: /modules/.*class file has wrong version (?<version>[4-9][0-9])\\.");

  private static final Pattern UNSUPPORTED_CLASS_VERSION_ERROR =
      Pattern.compile(
          "^(?<class>[^ ]*) has been compiled by a more recent version of the Java Runtime "
              + "\\(class file version (?<version>[4-9][0-9])\\.");

  /**
   * Sets up a BlazeJavaCompiler with the given plugins within the given context.
   *
   * @param context JavaCompiler's associated Context
   */
  @VisibleForTesting
  static void setupBlazeJavaCompiler(
      ImmutableList<BlazeJavaCompilerPlugin> plugins, Context context) {
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      plugin.initializeContext(context);
    }
    BlazeJavaCompiler.preRegister(context, plugins);
  }

  public static BlazeJavacResult compile(BlazeJavacArguments arguments) {

    List<String> javacArguments = arguments.javacOptions();
    try {
      processPluginArgs(
          arguments.plugins(), arguments.javacOptions(), arguments.blazeJavacOptions());
    } catch (CancelRequestException e) {
      return BlazeJavacResult.cancelled(e.getMessage());
    } catch (InvalidCommandLineException e) {
      return BlazeJavacResult.error(e.getMessage());
    }

    Optional<WerrorCustomOption> maybeWerrorCustom =
        arguments.blazeJavacOptions().stream()
            .filter(arg -> arg.startsWith("-Werror:"))
            .collect(toOptional())
            .map(WerrorCustomOption::create);

    Context context = new Context();
    BlazeJavacStatistics.preRegister(context);
    CacheFSInfo.preRegister(context);
    setupBlazeJavaCompiler(arguments.plugins(), context);
    BlazeJavacStatistics.Builder builder = context.get(BlazeJavacStatistics.Builder.class);

    Status status = Status.ERROR;
    StringWriter errOutput = new StringWriter();
    // TODO(cushon): where is this used when a diagnostic listener is registered? Consider removing
    // it and handling exceptions directly in callers.
    PrintWriter errWriter = new PrintWriter(errOutput);
    Listener diagnosticsBuilder =
        new Listener(arguments.failFast(), maybeWerrorCustom, context, arguments.workDir());

    // Initialize parts of context that the filemanager depends on
    context.put(DiagnosticListener.class, diagnosticsBuilder);
    Log.instance(context).setWriters(errWriter);
    Options options = Options.instance(context);
    options.put("-Xlint:path", "path");
    options.put("expandJarClassPaths", "false");

    try (ClassloaderMaskingFileManager fileManager = new ClassloaderMaskingFileManager(context)) {

      setLocations(fileManager, arguments);

      JavacTask task =
          JavacTool.create()
              .getTask(
                  errWriter,
                  fileManager,
                  diagnosticsBuilder,
                  javacArguments,
                  /* classes= */ ImmutableList.of(),
                  fileManager.getJavaFileObjectsFromPaths(arguments.sourceFiles()),
                  context);

      try {
        status = fromResult(((JavacTaskImpl) task).doCall());
      } catch (PropagatedException e) {
        throw e.getCause();
      }
    } catch (Exception t) {
      Throwable cause = t.getCause();
      if (cause instanceof CancelRequestException) {
        return BlazeJavacResult.cancelled(cause.getMessage());
      }
      Matcher matcher;
      if (cause instanceof UnsupportedClassVersionError
          && (matcher = UNSUPPORTED_CLASS_VERSION_ERROR.matcher(cause.getMessage())).find()) {
        // Java 8 corresponds to class file major version 52.
        int processorVersion = Integer.parseUnsignedInt(matcher.group("version")) - 44;
        errWriter.printf(
            "The Java %d runtime used to run javac is not recent enough to run the processor %s, "
                + "which has been compiled targeting Java %d. Either register a Java toolchain "
                + "with a newer java_runtime or, if this processor has been built with Bazel, "
                + "specify a lower --tool_java_language_version.%n",
            Runtime.version().feature(),
            matcher.group("class").replace('/', '.'),
            processorVersion);
      }
      t.printStackTrace(errWriter);
      status = Status.CRASH;
    }
    errWriter.flush();
    ImmutableList<FormattedDiagnostic> diagnostics = diagnosticsBuilder.build();

    diagnostics.stream()
        .map(d -> maybeGetJavaConfigurationError(arguments, d))
        .flatMap(Optional::stream)
        .findFirst()
        .ifPresent(errOutput::append);

    boolean werror =
        diagnostics.stream().anyMatch(d -> d.getCode().equals("compiler.err.warnings.and.werror"));
    if (status.equals(Status.OK) && diagnosticsBuilder.werror()) {
      errOutput.append("error: warnings found and -Werror specified\n");
      status = Status.ERROR;
      werror = true;
    }

    return BlazeJavacResult.createFullResult(
        status, filterDiagnostics(werror, diagnostics), errOutput.toString(), builder.build());
  }

  private static Status fromResult(Result result) {
    switch (result) {
      case OK:
        return Status.OK;
      case ERROR:
      case CMDERR:
      case SYSERR:
        return Status.ERROR;
      case ABNORMAL:
        return Status.CRASH;
    }
    throw new AssertionError(result);
  }

  private static final ImmutableSet<String> IGNORED_DIAGNOSTIC_CODES =
      ImmutableSet.of(
          "compiler.note.deprecated.filename",
          "compiler.note.deprecated.plural",
          "compiler.note.deprecated.recompile",
          "compiler.note.deprecated.filename.additional",
          "compiler.note.deprecated.plural.additional",
          "compiler.note.unchecked.filename",
          "compiler.note.unchecked.plural",
          "compiler.note.unchecked.recompile",
          "compiler.note.unchecked.filename.additional",
          "compiler.note.unchecked.plural.additional",
          "compiler.warn.sun.proprietary",
          // avoid warning spam when enabling processor options for an entire tree, only a subset
          // of which actually runs the processor
          "compiler.warn.proc.unmatched.processor.options",
          // don't want about v54 class files when running javac9 on JDK 10
          // TODO(cushon): remove after the next javac update
          "compiler.warn.big.major.version",
          // don't want about incompatible processor source versions when running javac9 on JDK 10
          // TODO(cushon): remove after the next javac update
          "compiler.warn.proc.processor.incompatible.source.version",
          // https://github.com/bazelbuild/bazel/issues/5985
          "compiler.warn.unknown.enum.constant",
          "compiler.warn.unknown.enum.constant.reason");

  private static ImmutableList<FormattedDiagnostic> filterDiagnostics(
      boolean werror, ImmutableList<FormattedDiagnostic> diagnostics) {
    return diagnostics.stream()
        .filter(d -> shouldReportDiagnostic(werror, d))
        // Print errors last to make them more visible.
        .sorted(comparing(FormattedDiagnostic::getKind).reversed())
        .collect(toImmutableList());
  }

  private static boolean shouldReportDiagnostic(boolean werror, FormattedDiagnostic diagnostic) {
    if (!IGNORED_DIAGNOSTIC_CODES.contains(diagnostic.getCode())) {
      return true;
    }
    // show compiler.warn.sun.proprietary if we're running with -Werror
    if (werror && diagnostic.getKind() != Diagnostic.Kind.NOTE) {
      return true;
    }
    return false;
  }

  private static Optional<String> maybeGetJavaConfigurationError(
      BlazeJavacArguments arguments, Diagnostic<?> diagnostic) {
    if (!diagnostic.getKind().equals(Diagnostic.Kind.ERROR)) {
      return Optional.empty();
    }
    Matcher matcher;
    if (!diagnostic.getCode().equals("compiler.err.cant.access")
        || arguments.system() == null
        || !(matcher = INCOMPATIBLE_SYSTEM_CLASS_PATH_ERROR.matcher(diagnostic.getMessage(ENGLISH)))
            .find()) {
      return Optional.empty();
    }
    // The output path is of the form $PRODUCT-out/$CPU-$MODE[-exec-...]/bin/...
    boolean isForTool = arguments.classOutput().subpath(1, 2).toString().contains("-exec-");
    // Java 8 corresponds to class file major version 52.
    int systemClasspathVersion = Integer.parseUnsignedInt(matcher.group("version")) - 44;
    return Optional.of(
        String.format(
            "error: [BazelJavaConfiguration] The Java %d runtime used to run javac is not recent "
                + "enough to compile for the Java %d runtime in %s. Either register a Java "
                + "toolchain with a newer java_runtime or specify a lower %s.\n",
            Runtime.version().feature(),
            systemClasspathVersion,
            arguments.system(),
            isForTool ? "--tool_java_runtime_version" : "--java_runtime_version"));
  }

  /** Processes Plugin-specific arguments and removes them from the args array. */
  @VisibleForTesting
  static void processPluginArgs(
      ImmutableList<BlazeJavaCompilerPlugin> plugins,
      ImmutableList<String> standardJavacopts,
      ImmutableList<String> blazeJavacopts)
      throws InvalidCommandLineException {
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      plugin.processArgs(standardJavacopts, blazeJavacopts);
    }
  }

  private static void setLocations(JavacFileManager fileManager, BlazeJavacArguments arguments) {
    try {
      fileManager.setLocationFromPaths(StandardLocation.CLASS_PATH, arguments.classPath());
      // modular dependencies must be on the module path, not the classpath
      fileManager.setLocationFromPaths(
          StandardLocation.locationFor("MODULE_PATH"), arguments.classPath());

      fileManager.setLocationFromPaths(
          StandardLocation.CLASS_OUTPUT, ImmutableList.of(arguments.classOutput()));
      if (arguments.nativeHeaderOutput() != null) {
        fileManager.setLocationFromPaths(
            StandardLocation.NATIVE_HEADER_OUTPUT,
            ImmutableList.of(arguments.nativeHeaderOutput()));
      }

      ImmutableList<Path> sourcePath = arguments.sourcePath();
      if (sourcePath.isEmpty()) {
        // javac expects a module-info-relative source path to be set when compiling modules,
        // otherwise it reports an error:
        // "file should be on source path, or on patch path for module"
        ImmutableList<Path> moduleInfos =
            arguments.sourceFiles().stream()
                .filter(f -> f.getFileName().toString().equals("module-info.java"))
                .collect(toImmutableList());
        if (moduleInfos.size() == 1) {
          sourcePath = ImmutableList.of(getOnlyElement(moduleInfos).toAbsolutePath().getParent());
        }
      }
      fileManager.setLocationFromPaths(StandardLocation.SOURCE_PATH, sourcePath);

      Path system = arguments.system();
      if (system != null) {
        fileManager.setLocationFromPaths(
            StandardLocation.locationFor("SYSTEM_MODULES"), ImmutableList.of(system));
      }
      // The bootclasspath may legitimately be empty if --release is being used.
      Collection<Path> bootClassPath = arguments.bootClassPath();
      if (!bootClassPath.isEmpty()) {
        fileManager.setLocationFromPaths(StandardLocation.PLATFORM_CLASS_PATH, bootClassPath);
      }
      fileManager.setLocationFromPaths(
          StandardLocation.ANNOTATION_PROCESSOR_PATH, arguments.processorPath());
      if (arguments.sourceOutput() != null) {
        fileManager.setLocationFromPaths(
            StandardLocation.SOURCE_OUTPUT, ImmutableList.of(arguments.sourceOutput()));
      }
    } catch (IOException e) {
      throw new IOError(e);
    }
  }

  /**
   * Ensure that classes that appear in the API between JavaBuilder and plugins are consistently
   * loaded by the same classloader. 'Plugins' here means both annotation processors and Error Prone
   * plugins. The annotation processor API is defined in the JDK and doesn't require any special
   * handling, since the versions in the system classloader will always be loaded preferentially.
   * For Error Prone plugins, we want to ensure that classes in the API are loaded from the same
   * classloader as JavaBuilder, but that other classes referenced by plugins are loaded from the
   * processor classpath to avoid plugins seeing stale versions of classes from the releases
   * JavaBuilder jar.
   */
  @Trusted
  private static class ClassloaderMaskingFileManager extends JavacFileManager {

    public ClassloaderMaskingFileManager(Context context) {
      super(context, true, UTF_8);
    }

    @Override
    protected ClassLoader getClassLoader(URL[] urls) {
      return new URLClassLoader(
          urls,
          new ClassLoader(ClassLoader.getPlatformClassLoader()) {
            @Override
            protected Class<?> findClass(String name) throws ClassNotFoundException {
              if (name.startsWith("com.google.errorprone.")
                  || name.startsWith("com.google.common.collect.")
                  || name.startsWith("com.google.common.base.")
                  || name.startsWith("com.google.common.regex.")
                  || name.startsWith("org.checkerframework.errorprone.dataflow.")
                  || name.startsWith("com.google.devtools.build.buildjar.javac.statistics.")) {
                return Class.forName(name);
              }
              throw new ClassNotFoundException(name);
            }
          });
    }
  }

  private BlazeJavacMain() {}
}
