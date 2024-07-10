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

package com.google.devtools.build.buildjar.javac.plugins.dependency;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.devtools.build.buildjar.javac.plugins.dependency.StrictJavaDepsPlugin.NonPlatformJar.Kind.FOR_JSPECIFY_FROM_PLATFORM;
import static com.google.devtools.build.buildjar.javac.plugins.dependency.StrictJavaDepsPlugin.NonPlatformJar.Kind.IN_CLASSPATH;
import static javax.tools.StandardLocation.CLASS_PATH;

import com.google.auto.value.AutoOneOf;
import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.buildjar.JarOwner;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule.StrictJavaDeps;
import com.google.devtools.build.buildjar.javac.statistics.BlazeJavacStatistics;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.devtools.build.lib.view.proto.Deps.Dependency;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.code.Kinds;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.resources.CompilerProperties.Errors;
import com.sun.tools.javac.resources.CompilerProperties.Warnings;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.TreeInfo;
import com.sun.tools.javac.tree.TreeScanner;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Log;
import com.sun.tools.javac.util.Log.WriterKind;
import com.sun.tools.javac.util.Name;
import com.sun.tools.javac.util.Names;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.jar.Attributes;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import javax.lang.model.element.AnnotationValue;
import javax.lang.model.util.SimpleAnnotationValueVisitor8;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.JavaFileObject.Kind;

/**
 * A plugin for BlazeJavaCompiler that checks for types referenced directly in the source, but
 * included through transitive dependencies. To get this information, we hook into the type
 * attribution phase of the BlazeJavaCompiler (thus the overhead is another tree scan with the
 * classic visitor). The constructor takes a map from jar names to target names, only for the jars
 * that come from transitive dependencies (Blaze computes this information).
 */
public final class StrictJavaDepsPlugin extends BlazeJavaCompilerPlugin {
  private static final Attributes.Name TARGET_LABEL = new Attributes.Name("Target-Label");
  private static final Attributes.Name INJECTING_RULE_KIND =
      new Attributes.Name("Injecting-Rule-Kind");

  private ImplicitDependencyExtractor implicitDependencyExtractor;
  private CheckingTreeScanner checkingTreeScanner;
  private final DependencyModule dependencyModule;

  /** Marks seen compilation toplevels and their import sections */
  private final Set<JCTree.JCCompilationUnit> toplevels;
  /** Marks seen ASTs */
  private final Set<JCTree> trees;
  /** Computed missing dependencies */
  private final Set<JarOwner> missingTargets;
  /** Strict deps diagnostics. */
  private final List<SjdDiagnostic> diagnostics;

  private PrintWriter errWriter;

  @AutoValue
  abstract static class SjdDiagnostic {
    abstract int pos();

    abstract String message();

    abstract JavaFileObject source();

    static SjdDiagnostic create(int pos, String message, JavaFileObject source) {
      return new AutoValue_StrictJavaDepsPlugin_SjdDiagnostic(pos, message, source);
    }
  }

  /**
   * On top of javac, we keep Blaze-specific information in the form of two maps. Both map jars
   * (exactly as they appear on the classpath) to target names, one is used for direct dependencies,
   * the other for the transitive dependencies.
   *
   * <p>This enables the detection of dependency issues. For instance, when a type com.Foo is
   * referenced in the source and it's coming from an indirect dependency, we emit a warning
   * flagging that dependency. Also, we can check whether the direct dependencies were actually
   * necessary, i.e. if their associated jars were used at all for looking up class definitions.
   */
  public StrictJavaDepsPlugin(DependencyModule dependencyModule) {
    this.dependencyModule = dependencyModule;
    toplevels = new HashSet<>();
    trees = new HashSet<>();
    missingTargets = new HashSet<>();
    diagnostics = new ArrayList<>();
  }

  @Override
  public void init(
      Context context,
      Log log,
      JavaCompiler compiler,
      BlazeJavacStatistics.Builder statisticsBuilder) {
    super.init(context, log, compiler, statisticsBuilder);
    errWriter = log.getWriter(WriterKind.ERROR);
    implicitDependencyExtractor =
        new ImplicitDependencyExtractor(
            dependencyModule.getImplicitDependenciesMap(),
            dependencyModule.getPlatformJars());
    checkingTreeScanner = context.get(CheckingTreeScanner.class);
    if (checkingTreeScanner == null) {
      checkingTreeScanner =
          new CheckingTreeScanner(
              dependencyModule,
              diagnostics,
              missingTargets,
              dependencyModule.getPlatformJars(),
              context.get(JavaFileManager.class),
              Names.instance(context));
      context.put(CheckingTreeScanner.class, checkingTreeScanner);
    }
  }

  /**
   * We want to make another pass over the AST and "type-check" the usage of direct/transitive
   * dependencies after the type attribution phase.
   */
  @Override
  public void postAttribute(Env<AttrContext> env) {
    JavaFileObject previousSource = checkingTreeScanner.source;
    try {
      if (isAnnotationProcessorExempt(env.toplevel)) {
        return;
      }
      checkingTreeScanner.source =
          env.enclClass.sym.sourcefile != null
              ? env.enclClass.sym.sourcefile
              : env.toplevel.sourcefile;
      if (trees.add(env.tree)) {
        checkingTreeScanner.scan(env.tree);
      }
      if (toplevels.add(env.toplevel)) {
        checkingTreeScanner.scan(env.toplevel.getImports());
        checkingTreeScanner.scan(env.toplevel.getPackage());
        dependencyModule.addPackage(env.toplevel.packge);
      }
    } finally {
      checkingTreeScanner.source = previousSource;
    }
  }

  @Override
  public void finish() {
    implicitDependencyExtractor.accumulate(context, checkingTreeScanner.getSeenClasses());

    for (SjdDiagnostic diagnostic : diagnostics) {
      JavaFileObject prev = log.useSource(diagnostic.source());
      try {
        switch (dependencyModule.getStrictJavaDeps()) {
          case ERROR:
            log.error(diagnostic.pos(), Errors.ProcMessager(diagnostic.message()));
            break;
          case WARN:
            log.warning(diagnostic.pos(), Warnings.ProcMessager(diagnostic.message()));
            break;
          case OFF: // continue below
        }
      } finally {
        log.useSource(prev);
      }
    }

    if (!missingTargets.isEmpty()) {
      String canonicalizedLabel =
          dependencyModule.getTargetLabel() == null
              ? null
              // we don't use the target mapping for the target, just the missing deps
              : canonicalizeTarget(dependencyModule.getTargetLabel());
      Set<JarOwner> canonicalizedMissing =
          missingTargets
              .stream()
              .filter(owner -> owner.label().isPresent())
              .sorted(Comparator.comparing((JarOwner owner) -> owner.label().get()))
              // for dependencies that are missing we canonicalize and remap the target so we don't
              // suggest private build labels.
              .map(owner -> owner.withLabel(owner.label().map(label -> canonicalizeTarget(label))))
              .collect(toImmutableSet());
      if (dependencyModule.getStrictJavaDeps() != StrictJavaDeps.OFF) {
        errWriter.print(
            dependencyModule.getFixMessage().get(canonicalizedMissing, canonicalizedLabel));
        dependencyModule.setHasMissingTargets();
      }
    }
  }

  /**
   * An AST visitor that implements our strict_java_deps checks. For now, it only emits warnings for
   * types loaded from jar files provided by transitive (indirect) dependencies. Each type is
   * considered only once, so at most one warning is generated for it.
   */
  private static class CheckingTreeScanner extends TreeScanner {

    private final ImmutableSet<Path> directJars;

    /** Strict deps diagnostics. */
    private final List<SjdDiagnostic> diagnostics;

    /** Missing targets */
    private final Set<JarOwner> missingTargets;

    /** Collect seen direct dependencies and their associated information */
    private final Map<Path, Deps.Dependency> directDependenciesMap;

    /** We only emit one warning/error per class symbol */
    private final Set<ClassSymbol> seenClasses = new HashSet<>();

    private final Set<JarOwner> seenTargets = new HashSet<>();

    private final Set<NonPlatformJar> seenStrictDepsViolatingJars = new HashSet<>();

    /** The set of jars on the compilation bootclasspath. */
    private final Set<Path> platformJars;

    private final JavaFileManager fileManager;

    /** The current source, for diagnostics. */
    private JavaFileObject source = null;

    /** Cache of classpath (not platform) jars in which given symbols can be found. */
    private final Map<ClassSymbol, Optional<Path>> classpathOnlyDepPaths = new HashMap<>();

    private final Name jspecifyAnnotationsPackage;

    public CheckingTreeScanner(
        DependencyModule dependencyModule,
        List<SjdDiagnostic> diagnostics,
        Set<JarOwner> missingTargets,
        Set<Path> platformJars,
        JavaFileManager fileManager,
        Names names) {
      this.directJars = dependencyModule.directJars();
      this.diagnostics = diagnostics;
      this.missingTargets = missingTargets;
      this.directDependenciesMap = dependencyModule.getExplicitDependenciesMap();
      this.platformJars = platformJars;
      this.fileManager = fileManager;
      this.jspecifyAnnotationsPackage = names.fromString("org.jspecify.annotations");
    }

    Set<ClassSymbol> getSeenClasses() {
      return seenClasses;
    }

    /** Checks an AST node denoting a class type against direct/transitive dependencies. */
    private void checkTypeLiteral(JCTree node, Symbol sym) {
      if (sym == null || sym.kind != Kinds.Kind.TYP) {
        return;
      }
      NonPlatformJar jar = getNonPlatformJar(sym.enclClass(), platformJars);

      // If this type symbol comes from a class file loaded from a jar, check
      // whether that jar was a direct dependency and error out otherwise.
      if (jar != null && seenClasses.add(sym.enclClass())) {
        collectExplicitDependency(jar, node, sym);
      }
    }

    /**
     * Marks the provided dependency as a direct/explicit dependency. Additionally, if
     * strict_java_deps is enabled, it emits a [strict] compiler warning/error.
     */
    private void collectExplicitDependency(NonPlatformJar jar, JCTree node, Symbol sym) {
      // Does it make sense to emit a warning/error for this pair of (type, owner)?
      // We want to emit only one error/warning per owner.
      if (!directJars.contains(jar.pathOrEmpty()) && seenStrictDepsViolatingJars.add(jar)) {
        // IO cost here is fine because we only hit this path for an explicit dependency
        // _not_ in the direct jars, i.e. an error
        JarOwner owner = readJarOwnerFromManifest(jar);
        if (seenTargets.add(owner)) {
          // owner is of the form "//label/of:rule <Aspect name>" where <Aspect name> is
          // optional.
          Optional<String> canonicalTargetName =
              owner.label().map(label -> canonicalizeTarget(label));
          missingTargets.add(owner);
          String toolInfo =
              owner.aspect().isPresent()
                  ? String.format(
                      "%s wrapped in %s", canonicalTargetName.get(), owner.aspect().get())
                  : canonicalTargetName.isPresent()
                      ? canonicalTargetName.get()
                      : owner.jar().toString();
          String used =
              sym.getSimpleName().contentEquals("package-info")
                  ? "package " + sym.getEnclosingElement()
                  : "type " + sym;
          String message =
              String.format(
                  "[strict] Using %s from an indirect dependency (TOOL_INFO: \"%s\").%s",
                  used, toolInfo, (owner.label().isPresent() ? " See command below **" : ""));
          diagnostics.add(SjdDiagnostic.create(node.pos, message, source));
        }
      }

      if (!directDependenciesMap.containsKey(jar.pathOrEmpty())) {
        // Also update the dependency proto
        Dependency dep =
            Dependency.newBuilder()
                // Path.toString uses the platform separator (`\` on Windows) which may not
                // match the format in params files (which currently always use `/`, see
                // bazelbuild/bazel#4108). JavaBuilder should always parse Path strings into
                // java.nio.file.Paths before comparing them.
                //
                // An empty path is OK in the cases we produce it. See readJarOwnerFromManifest.
                .setPath(jar.pathOrEmpty().toString())
                .setKind(Dependency.Kind.EXPLICIT)
                .build();
        directDependenciesMap.put(jar.pathOrEmpty(), dep);
      }
    }

    private JarOwner readJarOwnerFromManifest(NonPlatformJar jar) {
      if (jar.getKind() == FOR_JSPECIFY_FROM_PLATFORM) {
        return JSPECIFY_JAR_OWNER;
      }
      Path jarPath = jar.inClasspath();
      try (JarFile jarFile = new JarFile(jarPath.toFile())) {
        Manifest manifest = jarFile.getManifest();
        if (manifest == null) {
          return JarOwner.create(jarPath);
        }
        Attributes attributes = manifest.getMainAttributes();
        String label = (String) attributes.get(TARGET_LABEL);
        if (label == null) {
          return JarOwner.create(jarPath);
        }
        String injectingRuleKind = (String) attributes.get(INJECTING_RULE_KIND);
        return JarOwner.create(jarPath, label, Optional.ofNullable(injectingRuleKind));
      } catch (IOException e) {
        // This jar file pretty much has to exist, we just used it in the compiler. Throw unchecked.
        throw new UncheckedIOException(e);
      }
    }

    @Override
    public void visitMethodDef(JCTree.JCMethodDecl method) {
      if ((method.mods.flags & Flags.GENERATEDCONSTR) != 0) {
        // If this is the constructor for an anonymous inner class, refrain from checking the
        // compiler-generated method signature. Don't skip scanning the method body though, there
        // might have been an anonymous initializer which still needs to be checked.
        scan(method.body);
      } else {
        super.visitMethodDef(method);
      }
    }

    @Override
    public void visitVarDef(JCTree.JCVariableDecl variable) {
      scan(variable.mods);
      if (!declaredUsingVar(variable)) {
        scan(variable.vartype);
      }
      scan(variable.nameexpr);
      scan(variable.init);
    }

    private static boolean declaredUsingVar(JCTree.JCVariableDecl variableTree) {
      return DECLARED_USING_VAR.test(variableTree);
    }

    private static final Predicate<JCTree.JCVariableDecl> DECLARED_USING_VAR =
        getDeclaredUsingVar();

    private static Predicate<JCTree.JCVariableDecl> getDeclaredUsingVar() {
      Method method;
      try {
        method = JCTree.JCVariableDecl.class.getMethod("declaredUsingVar");
      } catch (ReflectiveOperationException e) {
        // The method in JCVariableDecl is only available in stock JDK 17.
        // There are no good options for earlier versions, short of looking at the source code and
        // re-parsing the variable declaration, which would be complicated and expensive. For now,
        // continue to enforce SJD on var for JDK < 17.
        return variableTree -> false;
      }
      return variableTree -> {
        try {
          return (boolean) method.invoke(variableTree);
        } catch (ReflectiveOperationException e) {
          throw new LinkageError(e.getMessage(), e);
        }
      };
    }

    /** Visits an identifier in the AST. We only care about type symbols. */
    @Override
    public void visitIdent(JCTree.JCIdent tree) {
      checkTypeLiteral(tree, tree.sym);
    }

    /**
     * Visits a field selection in the AST. We care because in some cases types may appear fully
     * qualified and only inside a field selection (e.g., "com.foo.Bar.X", we want to catch the
     * reference to Bar).
     */
    @Override
    public void visitSelect(JCTree.JCFieldAccess tree) {
      scan(tree.selected);
      checkTypeLiteral(tree, tree.sym);
    }

    @Override
    public void visitLambda(JCTree.JCLambda tree) {
      if (tree.paramKind != JCTree.JCLambda.ParameterKind.IMPLICIT) {
        // don't record type uses for implicitly typed lambda parameters
        scan(tree.params);
      }
      scan(tree.body);
    }

    @Override
    public void visitPackageDef(JCTree.JCPackageDecl tree) {
      scan(tree.annotations);
      checkTypeLiteral(tree, tree.packge.package_info);
    }

    /**
     * Returns the name of the <i>classpath</i> jar from which the given class symbol was loaded
     * (with an exception for the JSpecify annotations) or else {@code null}.
     *
     * <p>If the symbol came from the platform (i.e., system modules/bootclasspath), rather than
     * from the classpath, this method <i>usually</i> returns {@code null}. The exception is for
     * JSpecify-annotation symbols that are read from the platform: For such a symbol, this method
     * still returns the first <i>classpath</i> jar that contains the symbol, or, if no classpath
     * jar contains the symbol, it returns {@code FOR_JSPECIFY_FROM_PLATFORM}. (The calling code
     * later converts that to {@code JSPECIFY_JAR_OWNER}, which will lead to a strict-deps error,
     * since that jar clearly isn't a direct dependency.) In this way, we pretend that the
     * JSpecify-annotation symbols <i>aren't</i> part of the platform. That's important because in
     * fact they aren't part of it <i>at runtime</i> and so we want to force users of those classes
     * to declare a dependency on them.
     *
     * <p>This behavior is mildly unfortunate in the unusual situation that a project normally reads
     * a JSpecify-annotations class from an uber-jar, rather than from the normal JSpecify target.
     * In that case, we claim that the class is being loaded from the normal target. That is not the
     * target that the project's developers are likely to want. It's even possible that the class
     * isn't present on the reduced classpath but <i>would</i> be present (via the uber-jar) if only
     * we compiled with the full classpath. The full-classpath compilation would still produce a
     * strict-deps error, but it would produce one that recommends the correct jar/dependency. But
     * as this code is, we fail with a suggestion of the normal JSpecify target, and we may or may
     * not fall back to the full classpath.
     *
     * <p>OK, arguably it's unfortunate that <i>ever</i> we suggest that the normal JSpecify target
     * is on the classpath when it isn't really. However, the most common result of that is going to
     * be that we produce a more convenient error message. That convenience helps to offset any
     * confusion that we produce. Still, we won't introduce similar behavior for other classes
     * lightly.
     *
     * @param platformJars jars on javac's bootclasspath
     */
    private NonPlatformJar getNonPlatformJar(ClassSymbol classSymbol, Set<Path> platformJars) {
      if (classSymbol == null) {
        return null;
      }

      // Ignore symbols that appear in the sourcepath:
      if (haveSourceForSymbol(classSymbol)) {
        return null;
      }

      JavaFileObject classfile = classSymbol.classfile;

      Path path = ImplicitDependencyExtractor.getJarPath(classfile);
      // Filter out classes from the system modules and bootclasspath
      if (path == null || platformJars.contains(path)) {
        // ...except the JSpecify annotations, which we treat specially.
        if (classSymbol.packge().fullname.equals(jspecifyAnnotationsPackage)) {
          Path classpathJar = findLookingOnlyInClasspath(classSymbol);
          return classpathJar != null
              ? NonPlatformJar.forClasspathJar(classpathJar)
              : NonPlatformJar.FOR_JSPECIFY_FROM_PLATFORM;
        }
        return null;
      }

      return NonPlatformJar.forClasspathJar(path);
    }

    /**
     * Returns the first jar file in the classpath (not system modules, not bootclasspath) that
     * contains the given class or {@code null} if no such jar is available.
     */
    private Path findLookingOnlyInClasspath(ClassSymbol sym) {
      /*
       * computeIfAbsent doesn't cache null results, so we store Optional instances instead.
       *
       * In practice, that won't normally matter much: The only case in which our computation
       * function runs once per usage of a JSpecify-annotation class is the failing-build case—that
       * is, when the class is not on the classpath.
       */
      return classpathOnlyDepPaths
          .computeIfAbsent(
              sym,
              (unused) -> {
                try {
                  for (JavaFileObject file :
                      fileManager.list(
                          CLASS_PATH,
                          sym.packge().fullname.toString(),
                          ImmutableSet.of(Kind.CLASS),
                          false /* do not return classes in subpackages */)) {
                    /*
                     * The query above returns all classpath classes from the given package. We can
                     * imagine situations in which only *some* JSpecify annotations are present in a
                     * given classpath jar (an uber-jar with unused classes removed?), so we want to
                     * make sure that we found the class we want.
                     */
                    if (file.isNameCompatible(sym.getSimpleName().toString(), Kind.CLASS)) {
                      return Optional.of(ImplicitDependencyExtractor.getJarPath(file));
                    }
                  }
                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                }
                return Optional.empty();
              })
          .orElse(null);
    }
  }

  /**
   * Returns true if the compilation unit contains a single top-level class generated by an exempt
   * annotation processor (according to its {@link @Generated} annotation).
   *
   * <p>Annotation processors are expected to never generate more than one top level class, as
   * required by the style guide.
   */
  public boolean isAnnotationProcessorExempt(JCTree.JCCompilationUnit unit) {
    if (unit.getTypeDecls().size() != 1) {
      return false;
    }
    Symbol sym = TreeInfo.symbolFor(getOnlyElement(unit.getTypeDecls()));
    if (sym == null) {
      return false;
    }
    for (String value : getGeneratedBy(sym)) {
      if (dependencyModule.getExemptGenerators().contains(value)) {
        return true;
      }
    }
    return false;
  }

  private static ImmutableSet<String> getGeneratedBy(Symbol symbol) {
    ImmutableSet.Builder<String> suppressions = ImmutableSet.builder();
    symbol
        .getRawAttributes()
        .stream()
        .filter(
            a -> {
              Name name = a.type.tsym.getQualifiedName();
              return name.contentEquals("javax.annotation.Generated")
                  || name.contentEquals("javax.annotation.processing.Generated");
            })
        .flatMap(
            a ->
                a.getElementValues()
                    .entrySet()
                    .stream()
                    .filter(e -> e.getKey().getSimpleName().contentEquals("value"))
                    .map(e -> e.getValue()))
        .forEachOrdered(
            a ->
                a.accept(
                    new SimpleAnnotationValueVisitor8<Void, Void>() {
                      @Override
                      public Void visitString(String s, Void aVoid) {
                        suppressions.add(s);
                        return super.visitString(s, aVoid);
                      }

                      @Override
                      public Void visitArray(List<? extends AnnotationValue> vals, Void aVoid) {
                        vals.forEach(v -> v.accept(this, null));
                        return super.visitArray(vals, aVoid);
                      }
                    },
                    null));
    return suppressions.build();
  }

  /** Returns the canonical version of the target name. Package private for testing. */
  static String canonicalizeTarget(String target) {
    int colonIndex = target.indexOf(':');
    if (colonIndex == -1) {
      // No ':' in target, nothing to do.
      return target;
    }
    int lastSlash = target.lastIndexOf('/', colonIndex);
    if (lastSlash == -1) {
      // No '/' or target is actually a filename in label format, return unmodified.
      return target;
    }
    String packageName = target.substring(lastSlash + 1, colonIndex);
    String suffix = target.substring(colonIndex + 1);
    if (packageName.equals(suffix)) {
      // target ends in "/something:something", canonicalize.
      return target.substring(0, colonIndex);
    }
    return target;
  }

  /** Returns true if the given classSymbol corresponds to one of the sources being compiled. */
  private static boolean haveSourceForSymbol(ClassSymbol classSymbol) {
    if (classSymbol.sourcefile == null) {
      return false;
    }

    try {
      // The classreader uses metadata to populate the symbol's sourcefile with a fake file object.
      // Call getLastModified() to check if it's a real file:
      classSymbol.sourcefile.getLastModified();
    } catch (UnsupportedOperationException e) {
      return false;
    }

    return true;
  }

  @Override
  public boolean runOnAttributionErrors() {
    return true;
  }

  /**
   * Either a jar in the classpath or the well-known jar that contains the classes that are present
   * in the platform at compile time but not runtime.
   */
  @AutoOneOf(NonPlatformJar.Kind.class)
  abstract static class NonPlatformJar {
    enum Kind {
      IN_CLASSPATH,
      FOR_JSPECIFY_FROM_PLATFORM,
    }

    abstract Kind getKind();

    abstract Path inClasspath();

    abstract Placeholder forJspecifyFromPlatform();

    final Path pathOrEmpty() {
      return getKind() == IN_CLASSPATH ? inClasspath() : EMPTY_PATH;
    }

    static NonPlatformJar forClasspathJar(Path s) {
      return AutoOneOf_StrictJavaDepsPlugin_NonPlatformJar.inClasspath(s);
    }

    static final NonPlatformJar FOR_JSPECIFY_FROM_PLATFORM =
        AutoOneOf_StrictJavaDepsPlugin_NonPlatformJar.forJspecifyFromPlatform(Placeholder.INSTANCE);
  }

  enum Placeholder {
    INSTANCE
  }

  private static final Path EMPTY_PATH = Path.of("");

  /**
   * A special-purpose {@link JarOwner} instance that points to the main JSpecify target but is used
   * when the JSpecify annotations are instead read from the platform.
   *
   * <p>We use this instance to force users to add the explicit JSpecify dependency—again, even
   * though the annotations are present in the compile-time platform (i.e., bootclasspath or system
   * modules). We require users to add the dependency because the annotations are <i>not</i> present
   * in the <i>runtime</i> platform.
   *
   * <p>The {@link Path} argument that we pass to this instance doesn't matter because the build is
   * usually going to fail. (Or, if a strict-deps enforcement is disabled, the user can't reasonably
   * expect fully accurate dependency information, and our tools should be mostly resilient to an
   * empty path.)
   */
  private static final JarOwner JSPECIFY_JAR_OWNER =
      JarOwner.create(
          EMPTY_PATH, "//third_party/java/jspecify_annotations", /* aspect= */ Optional.empty());
}
