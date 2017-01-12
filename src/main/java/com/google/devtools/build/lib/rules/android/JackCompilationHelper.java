// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import javax.annotation.Nullable;

/**
 * Builds Jack actions for a Java or Android target.
 *
 * <p>Jack is the new Android toolchain which integrates proguard-compatible code minification
 * et al. and has an intermediate library format to front-load the dexing work.
 *
 * @see <a href="http://tools.android.com/tech-docs/jackandjill">Jack documentation</a>
 * @see JackLibraryProvider
 */
public final class JackCompilationHelper {

  private static final String PARTIAL_JACK_DIRECTORY = "_jill";

  private static final String JACK_DIRECTORY = "_jack";

  /** Filetype for the intermediate library created by Jack. */
  public static final FileType JACK_LIBRARY_TYPE = FileType.of(".jack");

  /** Flag to indicate that the next argument is a Jack property. */
  static final String JACK_PROPERTY = "-D";
  /** Flag to indicate that resource conflicts should be resolved by taking the first element. */
  static final String PROPERTY_KEEP_FIRST_RESOURCE = "jack.import.resource.policy=keep-first";
  /** Flag to indicate that type conflicts should be resolved by taking the first element. */
  static final String PROPERTY_KEEP_FIRST_TYPE = "jack.import.type.policy=keep-first";
  /** Flag to turn on/off sanity checks in Jack. */
  static final String SANITY_CHECKS = "--sanity-checks";
  /** Value of the sanity checks flag which disables sanity checks. */
  static final String SANITY_CHECKS_OFF = "off";
  /** Value of the sanity checks flag which enables sanity checks. */
  static final String SANITY_CHECKS_ON = "on";
  /** Flag to enable tolerant mode in Jill, for compiling special jars (e.g., bootclasspath). */
  static final String TOLERANT = "--tolerant";

  /** Flag to indicate the classpath of Jack libraries, separated by semicolons. */
  static final String CLASSPATH = "-cp";
  /** Flag to load a Jack library into the Jack compiler so it will be part of the output. */
  static final String IMPORT_JACK_LIBRARY = "--import";
  /** Flag to import a zip file of Java sources into a Jack library. */
  static final String IMPORT_SOURCE_ZIP = "--import-source-zip";
  /** Flag to import a single file into a Jack library's resources. */
  static final String IMPORT_RESOURCE_FILE = "--import-resource-file";
  /** Flag to add a zip file full of resources to the Jack library. */
  static final String IMPORT_RESOURCE_ZIP = "--import-resource-zip";
  /** Flag to add the names of annotation processors. */
  static final String PROCESSOR_NAMES = "--processor";
  /** Flag to add the classpath of annotation processors. */
  static final String PROCESSOR_CLASSPATH = "--processorpath";
  /** Flag to include a Proguard configuration. */
  static final String CONFIG_PROGUARD = "--config-proguard";
  /** Flag to set the multidex mode when compiling to dex with Jack. */
  static final String MULTI_DEX = "--multi-dex";
  /** Flag to specify the path to the main dex list in manual main dex mode. */
  static final String MAIN_DEX_LIST = "--main-dex-list";

  /** Flag indicating the filename Jill should output the converted jar to. */
  static final String JILL_OUTPUT = "--output";
  /** Flag to output a jack library. */
  static final String OUTPUT_JACK = "--output-jack";
  /** Flag to output a zip file containing dex files and resources for packaging. */
  static final String OUTPUT_DEX_ZIP = "--output-dex-zip";
  /** Name of the zip file containing the dex files and java resources for packaging. */
  static final String ZIP_OUTPUT_FILENAME = "classes.dex.zip";

  /** Rule context used to build and register actions. */
  private final RuleContext ruleContext;

  /** True to use Jack's internal sanity checks, trading speed for crash-on-bugs. */
  private final boolean useSanityChecks;
  /** True to make Jill more tolerant, when compiling special jars (e.g., bootclasspath) */
  private final boolean useTolerant;

  /** Binary used to extract resources from a jar file. */
  private final FilesToRunProvider resourceExtractorBinary;
  /** Binary used to build Jack libraries and dex files. */
  private final FilesToRunProvider jackBinary;
  /** Binary used to convert jars to Jack libraries. */
  private final FilesToRunProvider jillBinary;
  /**
   * Jack libraries containing Android/Java base classes.
   *
   * <p>These will be placed first on the classpath.
   */
  private final NestedSet<Artifact> baseClasspath;

  /** The destination for the Jack artifact to be created, or null to skip this. */
  @Nullable private final Artifact outputArtifact;

  /** Java files for the rule's Jack library. */
  private final ImmutableSet<Artifact> javaSources;
  /** Zip files of java sources for the rule's Jack library. */
  private final ImmutableSet<Artifact> sourceJars;

  /** Java resources for the rule's Jack library. */
  private final ImmutableMap<PathFragment, Artifact> resources;

  /** Jars that contain resources to be added to the Jack library. */
  private final NestedSet<Artifact> resourceJars;

  /** Jack libraries to be provided to depending rules on the classpath, from srcs and exports. */
  private final NestedSet<Artifact> exportedJacks;
  /**
   * Jar libraries to be provided to depending rules on the classpath, from srcs and exports.
   * These will be placed after the jack files from exportedJacks and before this rule's jack file.
   */
  private final ImmutableSet<Artifact> exportedJars;

  /** Jack libraries to be provided only to this rule on the classpath, from srcs and deps. */
  private final NestedSet<Artifact> classpathJacks;
  /**
   * Jar libraries to be provided only to this rule on the classpath, from srcs and deps.
   * These will be placed after the jack files from classpathJacks.
   */
  private final ImmutableSet<Artifact> classpathJars;

  /**
   * Jack libraries from dependency libraries to be included in dexing.
   * Does not include the library generated by this rule or any reached only through neverlink libs.
   */
  private final NestedSet<Artifact> dexJacks;
  /** Jar libraries from dependency libraries to be included in dexing. */
  private final ImmutableSet<Artifact> dexJars;

  /** Minimal state used only to give nice error messages in case of oopses. */
  private JackLibraryProvider alreadyCompiledLibrary;

  private boolean wasDexBuilt;

  /** The classpath to be used for annotation processors. */
  private final NestedSet<Artifact> processorClasspathJars;

  /** The names of classes to be used as annotation processors. */
  private final ImmutableSet<String> processorNames;

  /** Creates a new JackCompilationHelper. Called from {@link Builder#build()}. */
  private JackCompilationHelper(
      RuleContext ruleContext,
      boolean useSanityChecks,
      boolean useTolerant,
      FilesToRunProvider resourceExtractorBinary,
      FilesToRunProvider jackBinary,
      FilesToRunProvider jillBinary,
      NestedSet<Artifact> baseClasspath,
      @Nullable Artifact outputArtifact,
      ImmutableSet<Artifact> javaSources,
      ImmutableSet<Artifact> sourceJars,
      ImmutableMap<PathFragment, Artifact> resources,
      NestedSet<Artifact> resourceJars,
      NestedSet<Artifact> processorClasspathJars,
      ImmutableSet<String> processorNames,
      NestedSet<Artifact> exportedJacks,
      ImmutableSet<Artifact> exportedJars,
      NestedSet<Artifact> classpathJacks,
      ImmutableSet<Artifact> classpathJars,
      NestedSet<Artifact> dexJacks,
      ImmutableSet<Artifact> dexJars) {
    this.ruleContext = ruleContext;
    this.useSanityChecks = useSanityChecks;
    this.useTolerant = useTolerant;
    this.resourceExtractorBinary = resourceExtractorBinary;
    this.jackBinary = jackBinary;
    this.jillBinary = jillBinary;
    this.baseClasspath = baseClasspath;
    this.outputArtifact = outputArtifact;
    this.javaSources = javaSources;
    this.sourceJars = sourceJars;
    this.resources = resources;
    this.resourceJars = resourceJars;
    this.processorClasspathJars = processorClasspathJars;
    this.processorNames = processorNames;
    this.exportedJacks = exportedJacks;
    this.exportedJars = exportedJars;
    this.classpathJacks = classpathJacks;
    this.classpathJars = classpathJars;
    this.dexJacks = dexJacks;
    this.dexJars = dexJars;
  }

  /**
   * Builds one or more dex files from the jack libraries in the transitive closure of this rule.
   *
   * <p>This method should only be called once, as it will generate the same artifact each time.
   * It will fail if called a second time.
   *
   * @param multidexMode The multidex flag to send to Jack.
   * @param manualMainDexList Iff multidexMode is MANUAL_MAIN_DEX, an artifact representing the file
   *     with the list of class filenames which should go in the main dex. Else, absent.
   * @param proguardSpecs A collection of Proguard configuration files to be used to process the
   *     Jack libraries before building a dex out of them.
   * @returns A zip file containing the dex file(s) and Java resource(s) generated by Jack.
   */
  // TODO(bazel-team): this method (compile to jack library, compile all transitive jacks to dex)
  // may be too much overhead for manydex (dex per library) mode.
  // Instead, consider running jack --output-dex right on the source files, bypassing the
  // intermediate jack library format.
  public Artifact compileAsDex(
      MultidexMode multidexMode,
      Optional<Artifact> manualMainDexList,
      Collection<Artifact> proguardSpecs) {
    Preconditions.checkNotNull(multidexMode);
    Preconditions.checkNotNull(manualMainDexList);
    Preconditions.checkNotNull(proguardSpecs);
    Preconditions.checkArgument(
        multidexMode.isSupportedByJack(),
        "Multidex mode '%s' is not supported by Jack",
        multidexMode);
    Preconditions.checkArgument(
        manualMainDexList.isPresent() == (multidexMode == MultidexMode.MANUAL_MAIN_DEX),
        "The main dex list must be supplied if and only if the multidex mode is 'manual_main_dex'");
    Preconditions.checkState(!wasDexBuilt, "A dex file has already been built.");

    Artifact outputZip = AndroidBinary.getDxArtifact(ruleContext, ZIP_OUTPUT_FILENAME);

    NestedSet<Artifact> transitiveJackLibraries =
        compileAsLibrary().getTransitiveJackLibrariesToLink();
    CustomCommandLine.Builder builder =
        CustomCommandLine.builder()
            // Have jack double-check its behavior and crash rather than producing invalid output
            .add(SANITY_CHECKS)
            .add(useSanityChecks ? SANITY_CHECKS_ON : SANITY_CHECKS_OFF)
            // Have jack take the first match in the event of a class or resource name collision.
            .add(JACK_PROPERTY)
            .add(PROPERTY_KEEP_FIRST_RESOURCE)
            .add(JACK_PROPERTY)
            .add(PROPERTY_KEEP_FIRST_TYPE);

    for (Artifact jackLibrary : transitiveJackLibraries) {
      builder.addExecPath(IMPORT_JACK_LIBRARY, jackLibrary);
    }
    for (Artifact proguardSpec : proguardSpecs) {
      builder.addExecPath(CONFIG_PROGUARD, proguardSpec);
    }
    builder.add(MULTI_DEX).add(multidexMode.getJackFlagValue());
    if (manualMainDexList.isPresent()) {
      builder.addExecPath(MAIN_DEX_LIST, manualMainDexList.get());
    }
    builder.addExecPath(OUTPUT_DEX_ZIP, outputZip);
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setExecutable(jackBinary)
            .addTransitiveInputs(transitiveJackLibraries)
            .addInputs(proguardSpecs)
            .addInputs(manualMainDexList.asSet())
            .addOutput(outputZip)
            .setCommandLine(builder.build())
            .setProgressMessage("Dexing " + ruleContext.getLabel() + " with Jack")
            .setMnemonic("AndroidJackDex")
            .build(ruleContext));
    return outputZip;
  }

  /**
   * Constructs the actions to compile a jack library for a neverlink lib.
   *
   * @returns a {@link JackLibraryProvider} containing the resulting transitive jack libraries.
   */
  public JackLibraryProvider compileAsNeverlinkLibrary() {
    JackLibraryProvider nonNeverlink = compileAsLibrary();
    return JackLibraryProvider.create(
        /* transitiveJackLibrariesToLink */
        NestedSetBuilder.<Artifact>emptySet(Order.NAIVE_LINK_ORDER),
        nonNeverlink.getTransitiveJackClasspathLibraries());
  }

  /**
   * Constructs the actions to compile a jack library for a non-neverlink lib.
   *
   * @returns a {@link JackLibraryProvider} containing the resulting transitive jack libraries.
   */
  public JackLibraryProvider compileAsLibrary() {
    if (alreadyCompiledLibrary != null) {
      // Because the JackCompilationHelper is immutable, compileAsLibrary will always produce the
      // same result for the life of the helper. The resulting library may be needed by clients
      // which also need to build a dex, e.g., AndroidBinary.
      return alreadyCompiledLibrary;
    }
    Function<Artifact, Artifact> nonLibraryFileConverter =
        CacheBuilder.newBuilder()
            .initialCapacity(exportedJars.size() + classpathJars.size())
            .build(
                new CacheLoader<Artifact, Artifact>() {
                  @Override
                  public Artifact load(Artifact artifact) throws Exception {
                    if (JavaSemantics.JAR.matches(artifact.getFilename())) {
                      return postprocessPartialJackAndAddResources(
                          convertJarToPartialJack(artifact), extractResourcesFromJar(artifact));
                    } else if (JACK_LIBRARY_TYPE.matches(artifact.getFilename())) {
                      return artifact;
                    }
                    throw new AssertionError("Invalid type for library file: " + artifact);
                  }
                });

    NestedSet<Artifact> transitiveClasspath =
        new NestedSetBuilder<Artifact>(Order.NAIVE_LINK_ORDER)
            .addAll(Iterables.transform(classpathJars, nonLibraryFileConverter))
            .addTransitive(classpathJacks)
            .build();

    // The base classpath needs to be first in the set's iteration order.
    // Then any jars or jack files specified directly, then dependencies from providers.
    NestedSet<Artifact> classpath =
        new NestedSetBuilder<Artifact>(Order.NAIVE_LINK_ORDER)
            .addTransitive(baseClasspath)
            .addTransitive(transitiveClasspath)
            .build();

    NestedSetBuilder<Artifact> exports = new NestedSetBuilder<>(Order.NAIVE_LINK_ORDER);
    NestedSetBuilder<Artifact> dexContents = new NestedSetBuilder<>(Order.NAIVE_LINK_ORDER);

    if (outputArtifact != null) {
      if (javaSources.isEmpty() && sourceJars.isEmpty() && resources.isEmpty()
          && resourceJars.isEmpty()) {
        // We still have to create SOMETHING to fulfill the artifact, but man, screw it
        buildEmptyJackAction();
      } else {
        buildJackAction(javaSources, sourceJars, resources, resourceJars, classpath);
        exports.add(outputArtifact);
        dexContents.add(outputArtifact);
      }
    }

    // These need to be added now so that they can be after the outputArtifact (if present).
    exports
        .addAll(Iterables.transform(exportedJars, nonLibraryFileConverter))
        .addTransitive(exportedJacks)
        .addTransitive(transitiveClasspath);
    dexContents
        .addAll(Iterables.transform(dexJars, nonLibraryFileConverter))
        .addTransitive(dexJacks);

    alreadyCompiledLibrary = JackLibraryProvider.create(dexContents.build(), exports.build());
    return alreadyCompiledLibrary;
  }

  /**
   * Generates an action which converts the jar to partial Jack format and returns the Jack file.
   *
   * <p>Partial Jack format does not contain resources or pre-dex files.
   *
   * @see #postprocessPartialJackAndAddResources(Artifact,Artifact)
   */
  private Artifact convertJarToPartialJack(Artifact jar) {
    Artifact result = ruleContext.getUniqueDirectoryArtifact(
        PARTIAL_JACK_DIRECTORY,
        FileSystemUtils.replaceExtension(jar.getRootRelativePath(), ".jack"),
        ruleContext.getBinOrGenfilesDirectory());
    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setExecutable(jillBinary);
    if (useTolerant) {
      builder.addArgument(TOLERANT);
    }
    ruleContext.registerAction(
        builder
            .addArgument(JILL_OUTPUT)
            .addOutputArgument(result)
            .addInputArgument(jar)
            .setProgressMessage(
                "Converting " + jar.getExecPath().getBaseName() + " to Jack library with Jill")
            .setMnemonic("AndroidJill")
            .build(ruleContext));
    return result;
  }

  /**
   * Generates an action which creates a zip file from the contents of the input jar, filtering out
   * non-resource files and returning a zip file containing only resources.
   */
  private Artifact extractResourcesFromJar(Artifact jar) {
    Artifact result =  ruleContext.getUniqueDirectoryArtifact(
        PARTIAL_JACK_DIRECTORY,
        FileSystemUtils.replaceExtension(jar.getRootRelativePath(), "-resources.zip"),
        ruleContext.getBinOrGenfilesDirectory());

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setExecutable(resourceExtractorBinary)
            .addInputArgument(jar)
            .addOutputArgument(result)
            .setProgressMessage("Extracting resources from " + jar.getExecPath().getBaseName())
            .setMnemonic("AndroidJillResources")
            .build(ruleContext));
    return result;
  }

  /**
   * Generates an action to finish processing a partial Jack library generated by
   * {@link #convertJarToPartialJack(Artifact)} and add resources from
   * {@link #extractResourcesFromJar(Artifact)}, then returns the final library.
   */
  private Artifact postprocessPartialJackAndAddResources(
      Artifact partialJackLibrary, Artifact resources) {
    Artifact result = ruleContext.getUniqueDirectoryArtifact(
        JACK_DIRECTORY,
        partialJackLibrary.getRootRelativePath().relativeTo(
            ruleContext.getUniqueDirectory(PARTIAL_JACK_DIRECTORY)),
        ruleContext.getBinOrGenfilesDirectory());
    CustomCommandLine.Builder builder =
        CustomCommandLine.builder()
            // Have jack double-check its behavior and crash rather than producing invalid output
            .add(SANITY_CHECKS)
            .add(useSanityChecks ? SANITY_CHECKS_ON : SANITY_CHECKS_OFF)
            .addExecPath(IMPORT_JACK_LIBRARY, partialJackLibrary)
            .addExecPath(IMPORT_RESOURCE_ZIP, resources)
            .addExecPath(OUTPUT_JACK, result);
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setExecutable(jackBinary)
            .addInput(partialJackLibrary)
            .addInput(resources)
            .addOutput(result)
            .setCommandLine(builder.build())
            .setProgressMessage(
                "Processing " + partialJackLibrary.getExecPath().getBaseName() + " as Jack library")
            .setMnemonic("AndroidJillPostprocess")
            .build(ruleContext));
    return result;
  }

  /**
   * Creates an action to build an empty jack library given by outputArtifact.
   */
  private void buildEmptyJackAction() {
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setExecutable(jackBinary)
            .addArgument(OUTPUT_JACK)
            .addOutputArgument(outputArtifact)
            .setProgressMessage("Compiling " + ruleContext.getLabel() + " as Jack library")
            .setMnemonic("AndroidJackLibraryNull")
            .build(ruleContext));
  }

  /**
   * Creates an action to compile the given sources as a jack library given by outputArtifact.
   *
   * @param javaSources Iterable of .java files to compile using jack.
   * @param sourceJars Iterable of .srcjar files to unpack and compile using jack.
   * @param resources Mapping from library paths to resource files to be imported into the jack
   *     library.
   * @param classpathJackLibraries Libraries used for compilation.
   * @returns An artifact representing the combined jack library, or null if none was created.
   */
  private void buildJackAction(
      Iterable<Artifact> javaSources,
      Iterable<Artifact> sourceJars,
      Map<PathFragment, Artifact> resources,
      NestedSet<Artifact> resourceJars,
      NestedSet<Artifact> classpathJackLibraries) {
    CustomCommandLine.Builder builder =
        CustomCommandLine.builder()
            // Have jack double-check its behavior and crash rather than producing invalid output
            .add(SANITY_CHECKS)
            .add(useSanityChecks ? SANITY_CHECKS_ON : SANITY_CHECKS_OFF)
            .addExecPath(OUTPUT_JACK, outputArtifact)
            .addJoinExecPaths(CLASSPATH, ":", classpathJackLibraries);
    if (!processorNames.isEmpty()) {
      builder.add(PROCESSOR_NAMES).add(Joiner.on(',').join(processorNames));
    }
    if (!processorClasspathJars.isEmpty()) {
      builder.addJoinExecPaths(PROCESSOR_CLASSPATH, ":", processorClasspathJars);
    }
    for (Entry<PathFragment, Artifact> resource : resources.entrySet()) {
      builder.add(IMPORT_RESOURCE_FILE);
      // Splits paths at the appropriate root (java root, if present; source/genfiles/etc. if not).
      // The part of the path after the : is used as the path to the resource within the jack/apk,
      // while the part of the path before the : is the remainder of the path to the resource file.
      // In cases where the path to the file and the path within the jack/apk are the same,
      // such as when a source file is not under a java root, this prefix will be empty.
      PathFragment execPath = resource.getValue().getExecPath();
      PathFragment resourcePath = resource.getKey();
      if (execPath.equals(resourcePath)) {
        builder.addPaths(":%s", resourcePath);
      } else {
        // execPath must end with resourcePath in all cases
        PathFragment rootPrefix =
            execPath.subFragment(0, execPath.segmentCount() - resourcePath.segmentCount());
        builder.addPaths("%s:%s", rootPrefix, resourcePath);
      }
    }
    if (!resourceJars.isEmpty()) {
      builder.addJoinExecPaths(IMPORT_RESOURCE_ZIP, ":", resourceJars);
    }
    builder.addBeforeEachExecPath(IMPORT_SOURCE_ZIP, sourceJars).addExecPaths(javaSources);
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setExecutable(jackBinary)
            .addTransitiveInputs(classpathJackLibraries)
            .addOutput(outputArtifact)
            .addTransitiveInputs(processorClasspathJars)
            .addInputs(resources.values())
            .addTransitiveInputs(resourceJars)
            .addInputs(sourceJars)
            .addInputs(javaSources)
            .setCommandLine(builder.build())
            .setProgressMessage("Compiling " + ruleContext.getLabel() + " as Jack library")
            .setMnemonic("AndroidJackLibrary")
            .build(ruleContext));
  }

  /**
   * Builder for JackCompilationHelper to configure all of its tools and sources.
   */
  public static final class Builder {

    /** Rule context used to build and register actions. */
    @Nullable private RuleContext ruleContext;

    /** Whether to enable tolerant mode in Jill, e.g., when compiling a bootclasspath. */
    private boolean useTolerant;

    /** Binary used to extract resources from a jar file. */
    @Nullable private FilesToRunProvider resourceExtractorBinary;
    /** Binary used to build Jack libraries and dex files. */
    @Nullable private FilesToRunProvider jackBinary;
    /** Binary used to convert jars to Jack libraries. */
    @Nullable private FilesToRunProvider jillBinary;
    /**
     * Set of Jack libraries containing Android/Java base classes.
     *
     * <p>These will be placed first on the classpath.
     */
    @Nullable private NestedSet<Artifact> baseClasspath;

    /** The destination for the Jack artifact to be created. */
    @Nullable private Artifact outputArtifact;

    /** Java files for the rule's Jack library. */
    private final LinkedHashSet<Artifact> javaSources = new LinkedHashSet<>();
    /** Zip files of java sources for the rule's Jack library. */
    private final LinkedHashSet<Artifact> sourceJars = new LinkedHashSet<>();
    /** Map from paths within the Jack library to Java resources for the rule's Jack library. */
    private final LinkedHashMap<PathFragment, Artifact> resources = new LinkedHashMap<>();

    /** Set of resource jars that contain Java resources. */
    private final NestedSetBuilder<Artifact> resourceJars = NestedSetBuilder.stableOrder();

    /** Jack libraries to be provided to depending rules on the classpath, from srcs and exports. */
    private final NestedSetBuilder<Artifact> exportedJackLibraries =
        new NestedSetBuilder<>(Order.NAIVE_LINK_ORDER);
    /**
     * Jar libraries to be provided to depending rules on the classpath, from srcs and exports.
     * These will be placed after the jack files from exportedJacks and before this rule's jack
     * file.
     */
    private final LinkedHashSet<Artifact> exportedNonLibraryFiles = new LinkedHashSet<>();

    /** Jack libraries to be provided only to this rule on the classpath, from srcs and deps. */
    private final NestedSetBuilder<Artifact> classpathJackLibraries =
        new NestedSetBuilder<>(Order.NAIVE_LINK_ORDER);
    /**
     * Jar libraries to be provided only to this rule on the classpath, from srcs and deps.
     * These will be placed after the jack files from classpathJacks.
     */
    private final LinkedHashSet<Artifact> classpathNonLibraryFiles = new LinkedHashSet<>();

    /** The names of classes to be used as annotation processors. */
    private final LinkedHashSet<String> processorNames = new LinkedHashSet<>();

    /** Jar libraries to be provided as annotation processors' classpath, from plugin deps. */
    private final NestedSetBuilder<Artifact> processorClasspathJars =
        new NestedSetBuilder<>(Order.NAIVE_LINK_ORDER);

    /**
     * Jack libraries from dependency libraries to be included in dexing.
     * Does not include the library generated by this rule or any reached only through neverlink
     * libs.
     */
    private final NestedSetBuilder<Artifact> dexJacks =
        new NestedSetBuilder<>(Order.NAIVE_LINK_ORDER);
    /** Jar libraries from dependency libraries to be included in dexing. */
    private final LinkedHashSet<Artifact> dexJars = new LinkedHashSet<>();

    /**
     * Sets the rule context in which this compilation helper will operate.
     *
     * <p>The compilation tools will be loaded automatically from the appropriate attributes:
     * Jack as $jack, Jill as $jill, and the resource extractor as $resource_extractor.
     *
     * <p>Jack's sanity checks will be enabled or disabled according to the AndroidConfiguration
     * accessed through this context.
     */
    public JackCompilationHelper.Builder setRuleContext(RuleContext ruleContext) {
      this.ruleContext = Preconditions.checkNotNull(ruleContext);
      return this;
    }

    /**
     * Sets the artifact the final Jack library should be output to.
     *
     * <p>The artifact specified will always be generated, although it may be empty if there are no
     * sources.
     *
     * <p>This method must be called if any of addJavaSources, addSourceJars, or addResources is.
     */
    public JackCompilationHelper.Builder setOutputArtifact(Artifact outputArtifact) {
      this.outputArtifact = Preconditions.checkNotNull(outputArtifact);
      return this;
    }

    /**
     * Sets the Jack binary used to perform operations on Jack libraries.
     */
    public JackCompilationHelper.Builder setJackBinary(FilesToRunProvider jackBinary) {
      this.jackBinary = Preconditions.checkNotNull(jackBinary);
      return this;
    }

    /**
     * Sets the Jill binary used to translate jars to jack files.
     */
    public JackCompilationHelper.Builder setJillBinary(FilesToRunProvider jillBinary) {
      this.jillBinary = Preconditions.checkNotNull(jillBinary);
      return this;
    }

    /**
     * Sets the resource extractor binary used to extract resources from jars.
     */
    public JackCompilationHelper.Builder setResourceExtractorBinary(
        FilesToRunProvider resourceExtractorBinary) {
      this.resourceExtractorBinary = Preconditions.checkNotNull(resourceExtractorBinary);
      return this;
    }

    /**
     * Sets the base classpath, containing core classes (android.jar or Java bootclasspath).
     */
    public JackCompilationHelper.Builder setJackBaseClasspath(NestedSet<Artifact> baseClasspath) {
      this.baseClasspath = Preconditions.checkNotNull(baseClasspath);
      return this;
    }

    /**
     * Sets Jill to be tolerant, e.g., when translating a jar from the Java bootclasspath to jack.
     */
    public JackCompilationHelper.Builder setTolerant() {
      this.useTolerant = true;
      return this;
    }

    /**
     * Adds a collection of Java source files to be compiled by Jack.
     */
    public JackCompilationHelper.Builder addJavaSources(Collection<Artifact> javaSources) {
      this.javaSources.addAll(Preconditions.checkNotNull(javaSources));
      return this;
    }

    /**
     * Adds a collection of zip files containing Java sources to be compiled by Jack.
     */
    public JackCompilationHelper.Builder addSourceJars(Collection<Artifact> sourceJars) {
      this.sourceJars.addAll(Preconditions.checkNotNull(sourceJars));
      return this;
    }

    /**
     * Adds a collection of jar files to be converted to Jack libraries.
     *
     * <p>The Jack libraries created from these jar files will be used both as
     * dependencies on the classpath of this rule and exports to the classpath of depending rules,
     * as with jar files in the sources of a Java rule.
     * They will also be available to the compilation of the final dex file(s).
     * It has an identical effect as if these jars were specified in both deps and exports.
     */
    public JackCompilationHelper.Builder addCompiledJars(Collection<Artifact> compiledJars) {
      this.exportedNonLibraryFiles.addAll(Preconditions.checkNotNull(compiledJars));
      this.classpathNonLibraryFiles.addAll(compiledJars);
      this.dexJars.addAll(compiledJars);
      return this;
    }

    /**
     * Adds Java resources as a map keyed by the paths where they will be added to the Jack package
     * and the final APK. The resource path for an artifact must be a suffix of its exec path.
     */
    public JackCompilationHelper.Builder addResources(Map<PathFragment, Artifact> resources) {
      this.resources.putAll(Preconditions.checkNotNull(resources));
      return this;
    }

    public JackCompilationHelper.Builder addResourceJars(NestedSet<Artifact> resourceJars) {
      this.resourceJars.addTransitive(resourceJars);
      return this;
    }

    /**
     * Adds a set of class names which will be used as annotation processors.
     */
    public JackCompilationHelper.Builder addProcessorNames(Collection<String> processorNames) {
      this.processorNames.addAll(Preconditions.checkNotNull(processorNames));
      return this;
    }

    /**
     * Adds a set of jars which will be used as the classpath for annotation processors.
     */
    public JackCompilationHelper.Builder addProcessorClasspathJars(
        Iterable<Artifact> processorClasspathJars) {
      this.processorClasspathJars.addAll(Preconditions.checkNotNull(processorClasspathJars));
      return this;
    }

    /**
     * Adds a set of normal dependencies.
     *
     * <p>These dependencies will be considered direct dependencies of this rule,
     * and indirect dependencies of any rules depending on this one.
     * They will also be available to the compilation of the final dex file(s).
     */
    public JackCompilationHelper.Builder addDeps(
        Iterable<? extends TransitiveInfoCollection> deps) {
      return addClasspathDeps(deps).addRuntimeDeps(deps);
    }

    /**
     * Adds a set of dependencies which will be exported to Jack rules which depend on this one.
     *
     * <p>These dependencies will be considered direct dependencies of rules depending on this one,
     * but not of this rule itself, in line with Java rule semantics.
     * They will also be available to the compilation of the final dex file(s).
     */
    public JackCompilationHelper.Builder addExports(
        Iterable<? extends TransitiveInfoCollection> exports) {
      return addExportedDeps(exports).addRuntimeDeps(exports);
    }

    /**
     * Adds a set of dependencies which will be used in dexing.
     *
     * <p>Unless {@link #addClasspathDeps} or {@link #addExportedDeps} are also called,
     * runtimeDeps will not be provided on the classpath of this rule or rules which depend on it.
     */
    public JackCompilationHelper.Builder addRuntimeDeps(
        Iterable<? extends TransitiveInfoCollection> runtimeDeps) {
      for (TransitiveInfoCollection dep : Preconditions.checkNotNull(runtimeDeps)) {
        JackLibraryProvider jackLibraryProvider = dep.getProvider(JackLibraryProvider.class);
        if (jackLibraryProvider != null) {
          dexJacks.addTransitive(jackLibraryProvider.getTransitiveJackLibrariesToLink());
        } else {
          NestedSet<Artifact> filesToBuild = dep.getProvider(FileProvider.class).getFilesToBuild();
          for (Artifact file :
              FileType.filter(filesToBuild, JavaSemantics.JAR, JACK_LIBRARY_TYPE)) {
            dexJars.add(file);
          }
        }
      }
      return this;
    }

    /**
     * Adds a set of dependencies which will be placed on the classpath of this rule.
     *
     * <p>Unless {@link #addRuntimeDeps} is also called, classpathDeps will only be used for
     * compilation of this rule and will not be built into the final dex.
     *
     * @see #addDeps
     */
    public JackCompilationHelper.Builder addClasspathDeps(
        Iterable<? extends TransitiveInfoCollection> classpathDeps) {
      return addDependenciesInternal(
          Preconditions.checkNotNull(classpathDeps),
          classpathNonLibraryFiles,
          classpathJackLibraries);
    }

    /**
     * Adds a set of dependencies to be placed on the classpath of rules depending on this rule.
     *
     * <p>Unless {@link #addRuntimeDeps} is also called, exportedDeps will only be used for
     * compilation of depending rules and will not be built into the final dex.
     *
     * @see #addExports
     */
    public JackCompilationHelper.Builder addExportedDeps(
        Iterable<? extends TransitiveInfoCollection> exportedDeps) {
      return addDependenciesInternal(
          Preconditions.checkNotNull(exportedDeps), exportedNonLibraryFiles, exportedJackLibraries);
    }

    /**
     * Adds all libraries from deps to nonLibraryFiles and jackLibs based on their type.
     *
     * <p>Those exporting JackLibraryProvider have their jackClasspathLibraries added to jackLibs.
     * Others will have any jars or jacks in their filesToBuild added to nonLibraryFiles.
     *
     * <p>{@link #addRuntimeDeps} should also be called on deps for dependencies which will be built
     * into the final dex file(s).
     */
    private JackCompilationHelper.Builder addDependenciesInternal(
        Iterable<? extends TransitiveInfoCollection> deps,
        Collection<Artifact> nonLibraryFiles,
        NestedSetBuilder<Artifact> jackLibs) {
      for (TransitiveInfoCollection dep : deps) {
        JackLibraryProvider jackLibraryProvider = dep.getProvider(JackLibraryProvider.class);
        if (jackLibraryProvider != null) {
          jackLibs.addTransitive(jackLibraryProvider.getTransitiveJackClasspathLibraries());
        } else {
          NestedSet<Artifact> filesToBuild = dep.getProvider(FileProvider.class).getFilesToBuild();
          for (Artifact file :
              FileType.filter(filesToBuild, JavaSemantics.JAR, JACK_LIBRARY_TYPE)) {
            nonLibraryFiles.add(file);
          }
        }
      }
      return this;
    }

    /**
     * Constructs the JackCompilationHelper.
     *
     * <p>It's not recommended to call build() more than once, as the resulting
     * JackCompilationHelpers will attempt to generate the same actions.
     */
    public JackCompilationHelper build() {
      Preconditions.checkNotNull(ruleContext);

      boolean useSanityChecks =
          ruleContext
              .getFragment(AndroidConfiguration.class)
              .isJackSanityChecked();

      // It's okay not to have an outputArtifact if there is nothing to build.
      // e.g., if only translating jars with Jill, no final jack library will be created.
      // But if there is something to build, enforce that one has been specified.
      if (!javaSources.isEmpty() || !sourceJars.isEmpty() || !resources.isEmpty()) {
        Preconditions.checkNotNull(outputArtifact);
      }

      return new JackCompilationHelper(
          ruleContext,
          useSanityChecks,
          useTolerant,
          Preconditions.checkNotNull(resourceExtractorBinary),
          Preconditions.checkNotNull(jackBinary),
          Preconditions.checkNotNull(jillBinary),
          Preconditions.checkNotNull(baseClasspath),
          outputArtifact,
          ImmutableSet.copyOf(javaSources),
          ImmutableSet.copyOf(sourceJars),
          ImmutableMap.copyOf(resources),
          resourceJars.build(),
          processorClasspathJars.build(),
          ImmutableSet.copyOf(processorNames),
          exportedJackLibraries.build(),
          ImmutableSet.copyOf(exportedNonLibraryFiles),
          classpathJackLibraries.build(),
          ImmutableSet.copyOf(classpathNonLibraryFiles),
          dexJacks.build(),
          ImmutableSet.copyOf(dexJars));
    }
  }
}
