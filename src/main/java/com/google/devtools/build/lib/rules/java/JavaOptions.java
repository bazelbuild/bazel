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
package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.DefaultLabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsConverter;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.analysis.config.DefaultsPackage;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaOptimizationMode;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.StringSetConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Command-line options for building Java targets
 */
public class JavaOptions extends FragmentOptions {
  public static final String DEFAULT_LANGTOOLS = "//tools/jdk:langtools";

  /** Converter for --javabase and --host_javabase. */
  public static class JavabaseConverter implements Converter<String> {
    @Override
    public String convert(String input) throws OptionsParsingException {
      return input.isEmpty() ? Constants.TOOLS_REPOSITORY + "//tools/jdk:jdk" : input;
    }

    @Override
    public String getTypeDescription() {
      return "a string";
    }
  }

  /** Converter for --java_langtools. */
  public static class LangtoolsConverter extends DefaultLabelConverter {
    public LangtoolsConverter() {
      super(Constants.TOOLS_REPOSITORY + DEFAULT_LANGTOOLS);
    }
  }

  /** Converter for --javac_bootclasspath. */
  public static class BootclasspathConverter extends DefaultLabelConverter {
    public BootclasspathConverter() {
      super(Constants.TOOLS_REPOSITORY + "//tools/jdk:bootclasspath");
    }
  }

  /** Converter for --javac_extdir. */
  public static class ExtdirConverter extends DefaultLabelConverter {
    public ExtdirConverter() {
      super(Constants.TOOLS_REPOSITORY + "//tools/jdk:extdir");
    }
  }

  /** Converter for --javabuilder_top. */
  public static class JavaBuilderConverter extends DefaultLabelConverter {
    public JavaBuilderConverter() {
      super(Constants.TOOLS_REPOSITORY + "//tools/jdk:JavaBuilder_deploy.jar");
    }
  }


  /** Converter for --singlejar_top. */
  public static class SingleJarConverter extends DefaultLabelConverter {
    public SingleJarConverter() {
      super(Constants.TOOLS_REPOSITORY + "//tools/jdk:SingleJar_deploy.jar");
    }
  }

  /** Converter for --genclass_top. */
  public static class GenClassConverter extends DefaultLabelConverter {
    public GenClassConverter() {
      super(Constants.TOOLS_REPOSITORY + "//tools/jdk:GenClass_deploy.jar");
    }
  }

  /** Converter for --ijar_top. */
  public static class IjarConverter extends DefaultLabelConverter {
    public IjarConverter() {
      super(Constants.TOOLS_REPOSITORY + "//tools/jdk:ijar");
    }
  }

  /** Converter for --java_toolchain. */
  public static class JavaToolchainConverter extends DefaultLabelConverter {
    public JavaToolchainConverter() {
      super(Constants.TOOLS_REPOSITORY + "//tools/jdk:toolchain");
    }
  }

  /**
   * Converter for the --javawarn option.
   */
  public static class JavacWarnConverter extends StringSetConverter {
    public JavacWarnConverter() {
      super("all",
            "cast",
            "-cast",
            "deprecation",
            "-deprecation",
            "divzero",
            "-divzero",
            "empty",
            "-empty",
            "fallthrough",
            "-fallthrough",
            "finally",
            "-finally",
            "none",
            "options",
            "-options",
            "overrides",
            "-overrides",
            "path",
            "-path",
            "processing",
            "-processing",
            "rawtypes",
            "-rawtypes",
            "serial",
            "-serial",
            "unchecked",
            "-unchecked"
            );
    }
  }

  /**
   * Converter for the --experimental_java_classpath option.
   */
  public static class JavaClasspathModeConverter extends EnumConverter<JavaClasspathMode> {
    public JavaClasspathModeConverter() {
      super(JavaClasspathMode.class, "Java classpath reduction strategy");
    }
  }

  /**
   * Converter for the --java_optimization_mode option.
   */
  public static class JavaOptimizationModeConverter extends EnumConverter<JavaOptimizationMode> {
    public JavaOptimizationModeConverter() {
      super(JavaOptimizationMode.class, "Java optimization strategy");
    }
  }

  @Option(name = "javabase",
      defaultValue = "",
      converter = JavabaseConverter.class,
      category = "version",
      help = "JAVABASE used for the JDK invoked by Blaze. This is the "
          + "JAVABASE which will be used to execute external Java "
          + "commands.")
  public String javaBase;

  @Option(name = "java_toolchain",
      defaultValue = "",
      category = "version",
      converter = JavaToolchainConverter.class,
      help = "The name of the toolchain rule for Java.")
  public Label javaToolchain;

  @Option(name = "host_javabase",
      defaultValue = "",
      converter = JavabaseConverter.class,
      category = "version",
      help = "JAVABASE used for the host JDK. This is the JAVABASE which is used to execute "
           + " tools during a build.")
  public String hostJavaBase;

  @Option(name = "javacopt",
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      help = "Additional options to pass to javac.")
  public List<String> javacOpts;

  @Option(name = "jvmopt",
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      help = "Additional options to pass to the Java VM. These options will get added to the "
          + "VM startup options of each java_binary target.")
  public List<String> jvmOpts;

  @Option(name = "javawarn",
      converter = JavacWarnConverter.class,
      defaultValue = "",
      category = "flags",
      allowMultiple = true,
      help = "Additional javac warnings to enable when compiling Java source files.")
  public List<String> javaWarns;

  @Option(name = "use_ijars",
      defaultValue = "true",
      category = "strategy",
      help = "If enabled, this option causes Java compilation to use interface jars. "
          + "This will result in faster incremental compilation, "
          + "but error messages can be different.")
  public boolean useIjars;

  @Deprecated
  @Option(name = "use_src_ijars",
      defaultValue = "false",
      category = "undocumented",
      help = "No-op. Kept here for backwards compatibility.")
  public boolean useSourceIjars;

  @Deprecated
  @Option(name = "experimental_incremental_ijars",
      defaultValue = "false",
      category = "undocumented",
      help = "No-op. Kept here for backwards compatibility.")
  public boolean incrementalIjars;

  @Option(name = "java_deps",
      defaultValue = "true",
      category = "strategy",
      help = "Generate dependency information (for now, compile-time classpath) per Java target.")
  public boolean javaDeps;

  @Option(name = "experimental_java_deps",
      defaultValue = "false",
      category = "experimental",
      expansion = "--java_deps",
      deprecationWarning = "Use --java_deps instead")
  public boolean experimentalJavaDeps;

  @Option(name = "experimental_java_classpath",
      allowMultiple = false,
      defaultValue = "javabuilder",
      converter = JavaClasspathModeConverter.class,
      category = "semantics",
      help = "Enables reduced classpaths for Java compilations.")
  public JavaClasspathMode experimentalJavaClasspath;

  @Option(name = "java_debug",
      defaultValue = "null",
      category = "testing",
      expansion = {"--test_arg=--wrapper_script_flag=--debug", "--test_output=streamed",
                   "--test_strategy=exclusive", "--test_timeout=9999", "--nocache_test_results"},
      help = "Causes the Java virtual machine of a java test to wait for a connection from a "
      + "JDWP-compliant debugger (such as jdb) before starting the test. Implies "
      + "-test_output=streamed."
      )
  public Void javaTestDebug;

  @Option(name = "strict_java_deps",
      allowMultiple = false,
      defaultValue = "default",
      converter = StrictDepsConverter.class,
      category = "semantics",
      help = "If true, checks that a Java target explicitly declares all directly used "
          + "targets as dependencies.")
  public StrictDepsMode strictJavaDeps;

  @Option(name = "javabuilder_top",
      defaultValue = "",
      category = "version",
      converter = JavaBuilderConverter.class,
      help = "Label of the filegroup that contains the JavaBuilder jar.")
  public Label javaBuilderTop;

  @Option(name = "javabuilder_jvmopt",
      allowMultiple = true,
      defaultValue = "",
      category = "undocumented",
      help = "Additional options to pass to the JVM when invoking JavaBuilder.")
  public List<String> javaBuilderJvmOpts;

  @Option(name = "singlejar_top",
      defaultValue = "",
      category = "version",
      converter = SingleJarConverter.class,
      help = "Label of the filegroup that contains the SingleJar jar.")
  public Label singleJarTop;

  @Option(name = "genclass_top",
      defaultValue = "",
      category = "version",
      converter = GenClassConverter.class,
      help = "Label of the filegroup that contains the GenClass jar.")
  public Label genClassTop;

  @Option(name = "ijar_top",
      defaultValue = "",
      category = "version",
      converter = IjarConverter.class,
      help = "Label of the filegroup that contains the ijar binary.")
  public Label iJarTop;

  @Option(name = "java_langtools",
      defaultValue = "",
      category = "version",
      converter = LangtoolsConverter.class,
      help = "Label of the rule that produces the Java langtools jar.")
  public Label javaLangtoolsJar;

  @Option(name = "javac_bootclasspath",
      defaultValue = "",
      category = "version",
      converter = BootclasspathConverter.class,
      help = "Label of the rule that produces the bootclasspath jars for javac to use.")
  public Label javacBootclasspath;

  @Option(name = "javac_extdir",
      defaultValue = "",
      category = "version",
      converter = ExtdirConverter.class,
      help = "Label of the rule that produces the extdir for javac to use.")
  public Label javacExtdir;

  @Option(name = "java_launcher",
      defaultValue = "null",
      converter = LabelConverter.class,
      category = "semantics",
      help = "If enabled, a specific Java launcher is used. "
          + "The \"launcher\" attribute overrides this flag. ")
  public Label javaLauncher;

  @Option(name = "extra_proguard_specs",
      allowMultiple = true,
      defaultValue = "", // Ignored
      converter = LabelConverter.class,
      category = "undocumented",
      help = "Additional Proguard specs that will be used for all Proguard invocations.  Note that "
          + "using this option only has an effect when Proguard is used anyway.")
  public List<Label> extraProguardSpecs;

  @Option(name = "translations",
      defaultValue = "auto",
      category = "semantics",
      help = "Translate Java messages; bundle all translations into the jar "
          + "for each affected rule.")
  public TriState bundleTranslations;

  @Option(name = "message_translations",
      defaultValue = "",
      category = "semantics",
      allowMultiple = true,
      help = "The message translations used for translating messages in Java targets.")
  public List<String> translationTargets;

  @Option(name = "check_constraint",
      allowMultiple = true,
      defaultValue = "",
      category = "checking",
      help = "Check the listed constraint.")
  public List<String> checkedConstraints;

  @Option(name = "experimental_disable_jvm",
      defaultValue = "false",
      category = "undocumented",
      help = "Disables the Jvm configuration entirely.")
  public boolean disableJvm;

  @Option(name = "java_optimization_mode",
      defaultValue = "legacy",
      converter = JavaOptimizationModeConverter.class,
      category = "undocumented",
      help = "Applies desired link-time optimizations to Java binaries and tests.")
  public JavaOptimizationMode javaOptimizationMode;

  @Option(name = "legacy_bazel_java_test",
      defaultValue = "true",
      category = "undocumented",
      help = "Use the legacy mode of Bazel for java_test.")
  public boolean legacyBazelJavaTest;

  @Override
  public FragmentOptions getHost(boolean fallback) {
    JavaOptions host = (JavaOptions) getDefault();

    host.javaBase = hostJavaBase;
    host.jvmOpts = ImmutableList.of("-client", "-XX:ErrorFile=/dev/stderr");

    host.javacOpts = javacOpts;
    host.javaLangtoolsJar = javaLangtoolsJar;
    host.javacExtdir = javacExtdir;
    host.javaBuilderTop = javaBuilderTop;
    host.javaToolchain = javaToolchain;
    host.singleJarTop = singleJarTop;
    host.genClassTop = genClassTop;
    host.iJarTop = iJarTop;

    // Java builds often contain complicated code generators for which
    // incremental build performance is important.
    host.useIjars = useIjars;

    host.javaDeps = javaDeps;
    host.experimentalJavaClasspath = experimentalJavaClasspath;

    return host;
  }

  @Override
  public void addAllLabels(Multimap<String, Label> labelMap) {
    addOptionalLabel(labelMap, "jdk", javaBase);
    addOptionalLabel(labelMap, "jdk", hostJavaBase);
    if (javaLauncher != null) {
      labelMap.put("java_launcher", javaLauncher);
    }
    labelMap.put("javabuilder", javaBuilderTop);
    labelMap.put("singlejar", singleJarTop);
    labelMap.put("genclass", genClassTop);
    labelMap.put("ijar", iJarTop);
    labelMap.put("java_toolchain", javaToolchain);
    labelMap.putAll("translation", getTranslationLabels());
  }

  @Override
  public Map<String, Set<Label>> getDefaultsLabels(BuildConfiguration.Options commonOptions) {
    Set<Label> jdkLabels = new LinkedHashSet<>();
    DefaultsPackage.parseAndAdd(jdkLabels, javaBase);
    DefaultsPackage.parseAndAdd(jdkLabels, hostJavaBase);
    Map<String, Set<Label>> result = new HashMap<>();
    result.put("JDK", jdkLabels);
    result.put("JAVA_LANGTOOLS", ImmutableSet.of(javaLangtoolsJar));
    result.put("JAVAC_BOOTCLASSPATH", ImmutableSet.of(javacBootclasspath));
    result.put("JAVAC_EXTDIR", ImmutableSet.of(javacExtdir));
    result.put("JAVABUILDER", ImmutableSet.of(javaBuilderTop));
    result.put("SINGLEJAR", ImmutableSet.of(singleJarTop));
    result.put("GENCLASS", ImmutableSet.of(genClassTop));
    result.put("IJAR", ImmutableSet.of(iJarTop));
    result.put("JAVA_TOOLCHAIN", ImmutableSet.of(javaToolchain));

    return result;
  }

  private Set<Label> getTranslationLabels() {
    Set<Label> result = new LinkedHashSet<>();
    for (String s : translationTargets) {
      try {
        Label label = Label.parseAbsolute(s);
        result.add(label);
      } catch (LabelSyntaxException e) {
        // We ignore this exception here - it will cause an error message at a later time.
      }
    }
    return result;
  }
}
