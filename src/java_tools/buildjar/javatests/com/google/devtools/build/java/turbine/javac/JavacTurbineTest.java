// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine.javac;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.java.turbine.TurbineOptions;
import com.google.devtools.build.java.turbine.javac.JavacTurbine.Result;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.devtools.build.lib.view.proto.Deps.Dependency;

import com.sun.source.util.JavacTask;
import com.sun.tools.javac.api.ClientCodeWrapper.Trusted;
import com.sun.tools.javac.api.JavacTool;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.util.Context;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.util.Textifier;
import org.objectweb.asm.util.TraceClassVisitor;

import java.io.BufferedInputStream;
import java.io.IOError;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.JarOutputStream;

import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.TypeElement;
import javax.tools.FileObject;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardLocation;

/** Unit tests for {@link JavacTurbine}. */
@RunWith(JUnit4.class)
public class JavacTurbineTest {

  @Rule public TemporaryFolder temp = new TemporaryFolder();

  Path sourcedir;
  List<Path> sources;
  Path tempdir;
  Path output;
  Path outputDeps;

  TurbineOptions.Builder optionsBuilder = TurbineOptions.builder();

  @Before
  public void setUp() throws IOException {
    sourcedir = temp.newFolder().toPath();
    tempdir = temp.newFolder("_temp").toPath();
    output = temp.newFile("out.jar").toPath();
    outputDeps = temp.newFile("out.jdeps").toPath();

    sources = new ArrayList<>();

    optionsBuilder
        .setOutput(output.toString())
        .setTempDir(tempdir.toString())
        .addBootClassPathEntries(
            ImmutableList.copyOf(Splitter.on(':').split(System.getProperty("sun.boot.class.path"))))
        .setOutputDeps(outputDeps.toString())
        .setStrictJavaDeps("ERROR")
        .addAllJavacOpts(Arrays.asList("-source", "7", "-target", "7"))
        .setTargetLabel("//test")
        .setRuleKind("java_library");
  }

  private void addSourceLines(String path, String... lines) throws IOException {
    Path source = sourcedir.resolve(path);
    sources.add(source);
    Files.write(source, Arrays.asList(lines), StandardCharsets.UTF_8);
  }

  void compile() throws IOException {
    optionsBuilder.addSources(ImmutableList.copyOf(Iterables.transform(sources, TO_STRING)));
    try (JavacTurbine turbine = new JavacTurbine(optionsBuilder.build())) {
      assertThat(turbine.compile()).isEqualTo(Result.OK_WITH_REDUCED_CLASSPATH);
    }
  }

  private Map<String, byte[]> collectOutputs() throws IOException {
    return collectFiles(output);
  }

  @Test
  public void hello() throws Exception {
    addSourceLines(
        "Hello.java",
        "class Hello {",
        "  public static void main(String[] args) {",
        "    System.err.println(\"Hello World\");",
        "  }",
        "}");

    compile();

    Map<String, byte[]> outputs = collectOutputs();

    assertThat(outputs.keySet()).containsExactly("Hello.class");

    String text = textify(outputs.get("Hello.class"));
    String[] expected = {
      "// class version 51.0 (51)",
      "// access flags 0x20",
      "class Hello {",
      "",
      "",
      "  // access flags 0x0",
      "  <init>()V",
      "",
      "  // access flags 0x9",
      "  public static main([Ljava/lang/String;)V",
      "}",
      ""
    };
    assertThat(text).isEqualTo(Joiner.on('\n').join(expected));
  }

  // verify that FLOW is disabled, as if we had passed -relax
  // if it isn't we'd get an error about the missing return in f().
  @Test
  public void relax() throws Exception {
    addSourceLines("Hello.java", "class Hello {", "  int f() {}", "}");

    compile();

    Map<String, byte[]> outputs = collectOutputs();

    assertThat(outputs.keySet()).containsExactly("Hello.class");

    String text = textify(outputs.get("Hello.class"));
    String[] expected = {
      "// class version 51.0 (51)",
      "// access flags 0x20",
      "class Hello {",
      "",
      "",
      "  // access flags 0x0",
      "  <init>()V",
      "",
      "  // access flags 0x0",
      "  f()I",
      "}",
      ""
    };
    assertThat(text).isEqualTo(Joiner.on('\n').join(expected));
  }

  public @interface MyAnnotation {}

  /**
   * A sample annotation processor for testing.
   *
   * <p>Writes two output files (one source, one data) the very first round it's called. Used to
   * verify that annotation processor output is collected into the output jar.
   */
  @SupportedAnnotationTypes("MyAnnotation")
  public static class MyProcessor extends AbstractProcessor {

    @Override
    public SourceVersion getSupportedSourceVersion() {
      return SourceVersion.latest();
    }

    boolean first = true;

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
      if (!first) {
        // Write the output files exactly once to ensure we don't try to write the same file
        // twice or do work on the final round.
        return false;
      }
      if (roundEnv.getRootElements().isEmpty()) {
        return false;
      }
      first = false;
      Element element = roundEnv.getRootElements().iterator().next();
      try {
        JavaFileObject sourceFile = processingEnv.getFiler().createSourceFile("Generated", element);
        try (OutputStream os = sourceFile.openOutputStream()) {
          os.write("class Generated {}".getBytes(StandardCharsets.UTF_8));
        }
      } catch (IOException e) {
        throw new IOError(e);
      }
      try {
        FileObject file =
            processingEnv
                .getFiler()
                .createResource(StandardLocation.CLASS_OUTPUT, "com.foo", "hello.txt", element);
        try (OutputStream os = file.openOutputStream()) {
          os.write("hello".getBytes(StandardCharsets.UTF_8));
        }
      } catch (IOException e) {
        throw new IOError(e);
      }
      return false;
    }
  }

  @Test
  public void processing() throws Exception {
    addSourceLines("MyAnnotation.java", "public @interface MyAnnotation {}");
    addSourceLines(
        "Hello.java",
        "@MyAnnotation",
        "class Hello {",
        "  public static void main(String[] args) {",
        "    System.err.println(\"Hello World\");",
        "  }",
        "}");

    optionsBuilder.setProcessors(ImmutableList.of(MyProcessor.class.getName()));
    optionsBuilder.addProcessorPathEntries(
        ImmutableList.copyOf(Splitter.on(':').split(System.getProperty("java.class.path"))));
    optionsBuilder.addClassPathEntries(
        ImmutableList.copyOf(Splitter.on(':').split(System.getProperty("java.class.path"))));

    compile();

    Map<String, byte[]> outputs = collectOutputs();
    assertThat(outputs.keySet())
        .containsExactly(
            "Generated.class", "MyAnnotation.class", "Hello.class", "com/foo/hello.txt");

    {
      String text = textify(outputs.get("Generated.class"));
      String[] expected = {
        "// class version 51.0 (51)",
        "// access flags 0x20",
        "class Generated {",
        "",
        "",
        "  // access flags 0x0",
        "  <init>()V",
        "}",
        ""
      };
      assertThat(text).isEqualTo(Joiner.on('\n').join(expected));
    }

    // sanity-check that annotation processing doesn't interfere with stripping
    {
      String text = textify(outputs.get("Hello.class"));
      String[] expected = {
        "// class version 51.0 (51)",
        "// access flags 0x20",
        "class Hello {",
        "",
        "",
        "  @LMyAnnotation;() // invisible",
        "",
        "  // access flags 0x0",
        "  <init>()V",
        "",
        "  // access flags 0x9",
        "  public static main([Ljava/lang/String;)V",
        "}",
        ""
      };
      assertThat(text).isEqualTo(Joiner.on('\n').join(expected));
    }
  }

  static Map<String, byte[]> collectFiles(Path jar) throws IOException {
    Map<String, byte[]> files = new LinkedHashMap<>();
    try (JarFile jf = new JarFile(jar.toFile())) {
      Enumeration<JarEntry> entries = jf.entries();
      while (entries.hasMoreElements()) {
        JarEntry entry = entries.nextElement();
        files.put(entry.getName(), ByteStreams.toByteArray(jf.getInputStream(entry)));
      }
    }
    return files;
  }

  static String textify(byte[] bytes) {
    StringWriter sw = new StringWriter();
    ClassReader cr = new ClassReader(bytes);
    cr.accept(new TraceClassVisitor(null, new Textifier(), new PrintWriter(sw, true)), 0);
    return sw.toString();
  }

  private static final Function<Object, String> TO_STRING =
      new Function<Object, String>() {
        @Override
        public String apply(Object input) {
          return String.valueOf(input);
        }
      };

  @Test
  public void extractLanguageLevel() {
    ImmutableList.Builder<String> output = ImmutableList.builder();
    JavacTurbine.addLanguageLevel(
        output,
        Arrays.asList(
            "-g", "-source", "6", "-target", "6", "-Xlint:all", "-source", "7", "-target", "7"));
    assertThat(output.build())
        .containsExactly("-source", "6", "-target", "6", "-source", "7", "-target", "7")
        .inOrder();
  }

  @Test
  public void jdeps() throws Exception {

    Path libC = temp.newFile("libc.jar").toPath();
    compileLib(
        libC,
        Collections.<Path>emptyList(),
        Arrays.asList(new StringJavaFileObject("C.java", "interface C { String getString(); }")));

    Path libA = temp.newFile("liba.jar").toPath();
    compileLib(
        libA,
        Collections.singleton(libC),
        Arrays.asList(new StringJavaFileObject("A.java", "interface A { C getC(); }")));

    Path depsA =
        writedeps(
            "liba.jdeps",
            Deps.Dependencies.newBuilder()
                .setSuccess(true)
                .setRuleLabel("//lib:a")
                .addDependency(
                    Deps.Dependency.newBuilder()
                        .setPath(libC.toString())
                        .setKind(Deps.Dependency.Kind.EXPLICIT))
                .build());

    Path libB = temp.newFile("libb.jar").toPath();
    compileLib(
        libB,
        Collections.<Path>emptyList(),
        Arrays.asList(new StringJavaFileObject("B.java", "interface B {}")));

    optionsBuilder.addClassPathEntries(
        ImmutableList.of(libA.toString(), libB.toString(), libC.toString()));
    optionsBuilder.addAllDepsArtifacts(ImmutableList.of(depsA.toString()));
    optionsBuilder.addDirectJarToTarget(libA.toString(), "//lib:a");
    optionsBuilder.addDirectJarToTarget(libB.toString(), "//lib:b");
    optionsBuilder.addIndirectJarToTarget(libC.toString(), "//lib:c");
    optionsBuilder.setTargetLabel("//my:target");

    addSourceLines(
        "Hello.java",
        "class Hello {",
        "  public static A a = null;",
        "  public static String s = a.getC().getString();",
        "  public static void main(String[] args) {",
        "    B b = null;",
        "  }",
        "}");

    compile();

    Deps.Dependencies depsProto = getDeps();

    assertThat(depsProto.getSuccess()).isTrue();
    assertThat(depsProto.getRuleLabel()).isEqualTo("//my:target");
    assertThat(getEntries(depsProto))
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                libA.toString(), Deps.Dependency.Kind.EXPLICIT,
                libB.toString(), Deps.Dependency.Kind.IMPLICIT,
                libC.toString(), Deps.Dependency.Kind.IMPLICIT));
  }

  private Map<String, Deps.Dependency.Kind> getEntries(Deps.Dependencies deps) {
    Map<String, Deps.Dependency.Kind> result = new LinkedHashMap<>();
    for (Dependency dep : deps.getDependencyList()) {
      result.put(dep.getPath(), dep.getKind());
    }
    return result;
  }

  private Deps.Dependencies getDeps() throws IOError {
    Deps.Dependencies depsProto;
    try (BufferedInputStream in = new BufferedInputStream(Files.newInputStream(outputDeps))) {
      Deps.Dependencies.Builder builder = Deps.Dependencies.newBuilder();
      builder.mergeFrom(in);
      depsProto = builder.build();
    } catch (IOException e) {
      throw new IOError(e);
    }
    return depsProto;
  }

  private void compileLib(
      Path jar, Iterable<Path> classpath, Iterable<? extends JavaFileObject> units)
      throws IOException {
    final Path outdir = temp.newFolder().toPath();
    JavacFileManager fm = new JavacFileManager(new Context(), false, StandardCharsets.UTF_8);
    fm.setLocationFromPaths(StandardLocation.CLASS_OUTPUT, Collections.singleton(outdir));
    fm.setLocationFromPaths(StandardLocation.CLASS_PATH, classpath);
    List<String> options = Arrays.asList("-d", outdir.toString());
    JavacTool tool = JavacTool.create();

    JavacTask task =
        tool.getTask(new PrintWriter(System.err, true), fm, null, options, null, units);
    assertThat(task.call()).isTrue();

    try (JarOutputStream jos = new JarOutputStream(Files.newOutputStream(jar))) {
      Files.walkFileTree(
          outdir,
          new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
                throws IOException {
              JarEntry je = new JarEntry(outdir.relativize(path).toString());
              jos.putNextEntry(je);
              Files.copy(path, jos);
              return FileVisitResult.CONTINUE;
            }
          });
    }
  }

  @Trusted
  static class StringJavaFileObject extends SimpleJavaFileObject {
    private final String content;

    StringJavaFileObject(String name, String... lines) {
      super(URI.create(name), JavaFileObject.Kind.SOURCE);
      this.content = Joiner.on('\n').join(lines);
    }

    @Override
    public CharSequence getCharContent(boolean ignoreEncodingErrors) throws IOException {
      return content;
    }
  }

  @Test
  public void reducedClasspath() throws Exception {

    Path libD = temp.newFile("libd.jar").toPath();
    compileLib(
        libD,
        Collections.<Path>emptyList(),
        Arrays.asList(new StringJavaFileObject("D.java", "public class D {}")));

    Path libC = temp.newFile("libc.jar").toPath();
    compileLib(
        libC,
        Collections.singleton(libD),
        Arrays.asList(new StringJavaFileObject("C.java", "class C { static D d; }")));

    Path libB = temp.newFile("libb.jar").toPath();
    compileLib(
        libB,
        Arrays.asList(libC, libD),
        Arrays.asList(new StringJavaFileObject("B.java", "class B { static C c; }")));

    Path libA = temp.newFile("liba.jar").toPath();
    compileLib(
        libA,
        Arrays.asList(libB, libC, libD),
        Arrays.asList(new StringJavaFileObject("A.java", "class A { static B b; }")));
    Path depsA =
        writedeps(
            "liba.jdeps",
            Deps.Dependencies.newBuilder()
                .setSuccess(true)
                .setRuleLabel("//lib:a")
                .addDependency(
                    Deps.Dependency.newBuilder()
                        .setPath(libB.toString())
                        .setKind(Deps.Dependency.Kind.EXPLICIT))
                .build());

    optionsBuilder.addClassPathEntries(
        ImmutableList.of(libA.toString(), libB.toString(), libC.toString(), libD.toString()));
    optionsBuilder.addAllDepsArtifacts(ImmutableList.of(depsA.toString()));
    optionsBuilder.addDirectJarToTarget(libA.toString(), "//lib:a");
    optionsBuilder.addIndirectJarToTarget(libB.toString(), "//lib:b");
    optionsBuilder.addIndirectJarToTarget(libC.toString(), "//lib:c");
    optionsBuilder.addIndirectJarToTarget(libD.toString(), "//lib:d");
    optionsBuilder.setTargetLabel("//my:target");

    addSourceLines(
        "Hello.java",
        "class Hello {",
        "  public static A a = new A();",
        "  public static void main(String[] args) {",
        "    A a = null;",
        "    B b = null;",
        "    C c = null;",
        "    D d = null;",
        "  }",
        "}");

    optionsBuilder.addSources(ImmutableList.copyOf(Iterables.transform(sources, TO_STRING)));

    try (JavacTurbine turbine = new JavacTurbine(optionsBuilder.build())) {
      assertThat(turbine.compile()).isEqualTo(Result.OK_WITH_REDUCED_CLASSPATH);
      Context context = turbine.context;

      JavacFileManager fm = (JavacFileManager) context.get(JavaFileManager.class);
      assertThat(fm.getLocationAsPaths(StandardLocation.CLASS_PATH)).containsExactly(libA, libB);

      Deps.Dependencies depsProto = getDeps();

      assertThat(depsProto.getSuccess()).isTrue();
      assertThat(depsProto.getRuleLabel()).isEqualTo("//my:target");
      assertThat(getEntries(depsProto))
          .containsExactlyEntriesIn(
              ImmutableMap.of(
                  libA.toString(),
                  Deps.Dependency.Kind.EXPLICIT,
                  libB.toString(),
                  Deps.Dependency.Kind.IMPLICIT));
    }
  }

  Path writedeps(String name, Deps.Dependencies deps) throws IOException {
    Path path = temp.newFile(name).toPath();
    try (OutputStream os = Files.newOutputStream(path)) {
      deps.writeTo(os);
    }
    return path;
  }

  @Test
  public void reducedClasspathFallback() throws Exception {

    Path libD = temp.newFile("libd.jar").toPath();
    compileLib(
        libD,
        Collections.<Path>emptyList(),
        Arrays.asList(
            new StringJavaFileObject("D.java", "public class D { static final int CONST = 42; }")));

    Path libC = temp.newFile("libc.jar").toPath();
    compileLib(
        libC,
        Collections.singleton(libD),
        Arrays.asList(new StringJavaFileObject("C.java", "class C extends D {}")));

    Path libB = temp.newFile("libb.jar").toPath();
    compileLib(
        libB,
        Arrays.asList(libC, libD),
        Arrays.asList(new StringJavaFileObject("B.java", "class B extends C {}")));

    Path libA = temp.newFile("liba.jar").toPath();
    compileLib(
        libA,
        Arrays.asList(libB, libC, libD),
        Arrays.asList(new StringJavaFileObject("A.java", "class A extends B {}")));
    Path depsA =
        writedeps(
            "liba.jdeps",
            Deps.Dependencies.newBuilder()
                .setSuccess(true)
                .setRuleLabel("//lib:a")
                .addDependency(
                    Deps.Dependency.newBuilder()
                        .setPath(libB.toString())
                        .setKind(Deps.Dependency.Kind.EXPLICIT))
                .build());

    optionsBuilder.addClassPathEntries(
        ImmutableList.of(libA.toString(), libB.toString(), libC.toString(), libD.toString()));
    optionsBuilder.addAllDepsArtifacts(ImmutableList.of(depsA.toString()));
    optionsBuilder.addDirectJarToTarget(libA.toString(), "//lib:a");
    optionsBuilder.addIndirectJarToTarget(libB.toString(), "//lib:b");
    optionsBuilder.addIndirectJarToTarget(libC.toString(), "//lib:c");
    optionsBuilder.addIndirectJarToTarget(libD.toString(), "//lib:d");
    optionsBuilder.setTargetLabel("//my:target");

    addSourceLines(
        "Hello.java",
        "class Hello {",
        "  public static final int CONST = A.CONST;",
        "  public static void main(String[] args) {}",
        "}");

    optionsBuilder.addSources(ImmutableList.copyOf(Iterables.transform(sources, TO_STRING)));

    try (JavacTurbine turbine = new JavacTurbine(optionsBuilder.build())) {
      assertThat(turbine.compile()).isEqualTo(Result.OK_WITH_FULL_CLASSPATH);
      Context context = turbine.context;

      JavacFileManager fm = (JavacFileManager) context.get(JavaFileManager.class);
      assertThat(fm.getLocationAsPaths(StandardLocation.CLASS_PATH))
          .containsExactly(libA, libB, libC, libD);

      Deps.Dependencies depsProto = getDeps();

      assertThat(depsProto.getSuccess()).isTrue();
      assertThat(depsProto.getRuleLabel()).isEqualTo("//my:target");
      assertThat(getEntries(depsProto))
          .containsExactlyEntriesIn(
              ImmutableMap.of(
                  libA.toString(), Deps.Dependency.Kind.EXPLICIT,
                  libB.toString(), Deps.Dependency.Kind.IMPLICIT,
                  libC.toString(), Deps.Dependency.Kind.IMPLICIT,
                  libD.toString(), Deps.Dependency.Kind.IMPLICIT));
    }
  }

  @Test
  public void constants() throws Exception {
    addSourceLines(
        "Const.java",
        "class Const {",
        "  public static final int A = 42;",
        "  public static final int B = 42 + 42;",
        "  public static final int C = new Integer(42);",
        "  public static final int D = 42 + new Integer(42);",
        "  public static final Integer E = 42;",
        "  public static final String F = \"42\";",
        "  public static final java.lang.String G = \"42\";",
        "}");

    compile();

    Map<String, byte[]> outputs = collectOutputs();

    assertThat(outputs.keySet()).containsExactly("Const.class");

    String text = textify(outputs.get("Const.class"));
    String[] expected = {
      "// class version 51.0 (51)",
      "// access flags 0x20",
      "class Const {",
      "",
      "",
      "  // access flags 0x19",
      "  public final static I A = 42",
      "",
      "  // access flags 0x19",
      "  public final static I B = 84",
      "",
      "  // access flags 0x19",
      "  public final static I C",
      "",
      "  // access flags 0x19",
      "  public final static I D",
      "",
      "  // access flags 0x19",
      "  public final static Ljava/lang/Integer; E",
      "",
      "  // access flags 0x19",
      "  public final static Ljava/lang/String; F = \"42\"",
      "",
      "  // access flags 0x19",
      "  public final static Ljava/lang/String; G = \"42\"",
      "",
      "  // access flags 0x0",
      "  <init>()V",
      "}",
      "",
    };
    assertThat(text).isEqualTo(Joiner.on('\n').join(expected));
  }

  @Test
  public void constantsEnum() throws Exception {
    addSourceLines(
        "TheEnum.java",
        // TODO(cushon): fix google-java-format's handling of lists of string literals
        "public enum TheEnum {",
        "  ONE, TWO, THREE;",
        "}");

    compile();
    Map<String, byte[]> outputs = collectOutputs();
    // just don't crash; enum constants need to be preserved
    assertThat(outputs.keySet()).containsExactly("TheEnum.class");

    String text = textify(outputs.get("TheEnum.class"));
    String[] expected = {
      "// class version 51.0 (51)",
      "// access flags 0x4031",
      "// signature Ljava/lang/Enum<LTheEnum;>;",
      "// declaration: TheEnum extends java.lang.Enum<TheEnum>",
      "public final enum TheEnum extends java/lang/Enum  {",
      "",
      "",
      "  // access flags 0x4019",
      "  public final static enum LTheEnum; ONE",
      "",
      "  // access flags 0x4019",
      "  public final static enum LTheEnum; TWO",
      "",
      "  // access flags 0x4019",
      "  public final static enum LTheEnum; THREE",
      "",
      "  // access flags 0x101A",
      "  private final static synthetic [LTheEnum; $VALUES",
      "",
      "  // access flags 0x9",
      "  public static values()[LTheEnum;",
      "",
      "  // access flags 0x9",
      "  public static valueOf(Ljava/lang/String;)LTheEnum;",
      "",
      "  // access flags 0x2",
      "  // signature ()V",
      "  // declaration: void <init>()",
      "  private <init>(Ljava/lang/String;I)V",
      "",
      "  // access flags 0x8",
      "  static <clinit>()V",
      "}",
      ""
    };
    assertThat(text).isEqualTo(Joiner.on('\n').join(expected));
  }

  /**
   * A sample annotation processor for testing.
   *
   * <p>Writes an output file that isn't valid UTF-8 to test handling of encoding errors.
   */
  @SupportedAnnotationTypes("MyAnnotation")
  public static class MyBadEncodingProcessor extends AbstractProcessor {

    @Override
    public SourceVersion getSupportedSourceVersion() {
      return SourceVersion.latest();
    }

    boolean first = true;

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
      if (!first) {
        return false;
      }
      if (roundEnv.getRootElements().isEmpty()) {
        return false;
      }
      first = false;
      Element element = roundEnv.getRootElements().iterator().next();
      try {
        JavaFileObject sourceFile = processingEnv.getFiler().createSourceFile("Generated", element);
        try (OutputStream os = sourceFile.openOutputStream()) {
          os.write(
              "class Generated { public static String x = \"".getBytes(StandardCharsets.UTF_8));
          os.write(0xc2); // write an unpaired surrogate
          os.write("\";}}".getBytes(StandardCharsets.UTF_8));
        }
      } catch (IOException e) {
        throw new IOError(e);
      }
      return false;
    }
  }

  @Test
  public void badEncoding() throws Exception {
    addSourceLines("MyAnnotation.java", "public @interface MyAnnotation {}");
    addSourceLines(
        "Hello.java",
        "@MyAnnotation",
        "class Hello {",
        "  public static void main(String[] args) {",
        "    System.err.println(\"Hello World\");",
        "  }",
        "}");

    optionsBuilder.setProcessors(ImmutableList.of(MyBadEncodingProcessor.class.getName()));
    optionsBuilder.addProcessorPathEntries(
        ImmutableList.copyOf(Splitter.on(':').split(System.getProperty("java.class.path"))));
    optionsBuilder.addClassPathEntries(
        ImmutableList.copyOf(Splitter.on(':').split(System.getProperty("java.class.path"))));

    optionsBuilder.addSources(ImmutableList.copyOf(Iterables.transform(sources, TO_STRING)));
    try (StringWriter sw = new StringWriter();
        JavacTurbine turbine =
            new JavacTurbine(new PrintWriter(sw, true), optionsBuilder.build())) {
      Result result = turbine.compile();
      assertThat(result).isEqualTo(Result.ERROR);
      assertThat(sw.toString()).contains("error reading");
    }
  }
}
