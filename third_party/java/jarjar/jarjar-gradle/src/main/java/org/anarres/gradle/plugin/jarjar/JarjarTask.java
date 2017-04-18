/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.anarres.gradle.plugin.jarjar;

import com.tonicsystems.jarjar.classpath.ClassPath;
import com.tonicsystems.jarjar.transform.JarTransformer;
import com.tonicsystems.jarjar.transform.config.ClassKeepTransitive;
import com.tonicsystems.jarjar.transform.config.ClassDelete;
import com.tonicsystems.jarjar.transform.config.ClassRename;
import com.tonicsystems.jarjar.transform.jar.DefaultJarProcessor;
import groovy.lang.Closure;
import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nonnull;
import org.apache.oro.text.GlobCompiler;
import org.apache.oro.text.regex.MalformedPatternException;
import org.apache.oro.text.regex.Pattern;
import org.apache.oro.text.regex.Perl5Matcher;
import org.gradle.api.artifacts.Configuration;
import org.gradle.api.artifacts.Dependency;
import org.gradle.api.artifacts.dsl.DependencyHandler;
import org.gradle.api.file.ConfigurableFileCollection;
import org.gradle.api.file.FileCollection;
import org.gradle.api.internal.ConventionTask;
import org.gradle.api.specs.Spec;
import org.gradle.api.tasks.InputFiles;
import org.gradle.api.tasks.OutputFile;
import org.gradle.api.tasks.OutputFiles;
import org.gradle.api.tasks.TaskAction;
import org.gradle.api.tasks.TaskOutputs;

/**
 *
 * @author shevek
 */
public class JarjarTask extends ConventionTask {

    private class FilterSpec implements Spec<File> {

        private final String message;
        private final Iterable<? extends Pattern> patterns;
        private final boolean result;

        public FilterSpec(@Nonnull String message, @Nonnull Iterable<? extends Pattern> patterns, boolean result) {
            this.message = message;
            this.patterns = patterns;
            this.result = result;
        }

        @Override
        public boolean isSatisfiedBy(File t) {
            if (matchesAny(patterns, t.getName())) {
                getLogger().info(message + " " + t);
                return result;
            }
            return !result;
        }

        @Override
        public String toString() {
            return getClass().getSimpleName() + "(patterns=" + patterns + ")";
        }
    }

    private static final Perl5Matcher globMatcher = new Perl5Matcher();

    private static boolean matchesAny(@Nonnull Iterable<? extends Pattern> patterns, @Nonnull String text) {
        for (Pattern pattern : patterns) {
            if (globMatcher.matches(text, pattern)) {
                return true;
            }
        }
        return false;
    }

    @Nonnull
    private static Iterable<Pattern> toPatterns(@Nonnull Iterable<? extends String>... patterns) throws MalformedPatternException {
        GlobCompiler compiler = new GlobCompiler();
        List<Pattern> out = new ArrayList<Pattern>();
        for (Iterable<? extends String> in : patterns)
            for (String pattern : in)
                out.add(compiler.compile(pattern));
        return out;
    }

    private final ConfigurableFileCollection sourceFiles;
    private final Set<String> archiveBypasses = new HashSet<String>();
    private final Set<String> archiveExcludes = new HashSet<String>();
    private File destinationDir;
    private String destinationName;

    private final DefaultJarProcessor processor = new DefaultJarProcessor();

    public JarjarTask() {
        sourceFiles = getProject().files();
    }

    @InputFiles
    public FileCollection getSourceFiles() {
        return sourceFiles;
    }

    /**
     * Returns the directory where the archive is generated into.
     *
     * @return the directory
     */
    public File getDestinationDir() {
        File out = destinationDir;
        if (out == null)
            out = new File(getProject().getBuildDir(), "jarjar");
        return out;
    }

    public void setDestinationDir(File destinationDir) {
        this.destinationDir = destinationDir;
    }

    /**
     * Returns the file name of the generated archive.
     *
     * @return the name
     */
    public String getDestinationName() {
        String out = destinationName;
        if (out == null)
            out = getName() + ".jar";
        return out;
    }

    public void setDestinationName(String destinationName) {
        this.destinationName = destinationName;
    }

    /**
     * The path where the archive is constructed.
     * The path is simply the {@code destinationDir} plus the {@code destinationName}.
     *
     * @return a File object with the path to the archive
     */
    @OutputFile
    public File getDestinationPath() {
        return new File(getDestinationDir(), getDestinationName());
    }

    @OutputFiles
    public FileCollection getBypassedArchives() throws MalformedPatternException {
        return sourceFiles.filter(new FilterSpec("Bypassing archive", toPatterns(archiveBypasses), true));
    }

    /**
     * Processes a FileCollection, which may be simple, a {@link Configuration},
     * or derived from a {@link TaskOutputs}.
     *
     * @param files The input FileCollection to consume.
     */
    public void from(@Nonnull FileCollection files) {
        sourceFiles.from(files);
    }

    /**
     * Processes a Dependency directly, which may be derived from
     * {@link DependencyHandler#create(java.lang.Object)},
     * {@link DependencyHandler#project(java.util.Map)},
     * {@link DependencyHandler#module(java.lang.Object)},
     * {@link DependencyHandler#gradleApi()}, etc.
     *
     * @param dependency The dependency to process.
     */
    public void from(@Nonnull Dependency dependency) {
        Configuration configuration = getProject().getConfigurations().detachedConfiguration(dependency);
        from(configuration);
    }

    /**
     * Processes a dependency specified by name.
     *
     * @param dependencyNotation The dependency, in a notation described in {@link DependencyHandler}.
     * @param configClosure The closure to use to configure the dependency.
     * @see DependencyHandler
     */
    public void from(@Nonnull String dependencyNotation, Closure configClosure) {
        from(getProject().getDependencies().create(dependencyNotation, configClosure));
    }

    /**
     * Processes a dependency specified by name.
     *
     * @param dependencyNotation The dependency, in a notation described in {@link DependencyHandler}.
     */
    public void from(@Nonnull String dependencyNotation) {
        from(getProject().getDependencies().create(dependencyNotation));
    }

    public void archiveBypass(@Nonnull String pattern) throws MalformedPatternException {
        archiveBypasses.add(pattern);
    }

    public void archiveExclude(@Nonnull String pattern) throws MalformedPatternException {
        archiveExcludes.add(pattern);
    }

    public void classRename(@Nonnull String pattern, @Nonnull String replacement) {
        processor.addClassRename(new ClassRename(pattern, replacement));
    }

    public void classDelete(@Nonnull String pattern) {
        processor.addClassDelete(new ClassDelete(pattern));
    }

    public void classClosureRoot(@Nonnull String pattern) {
        processor.addClassKeepTransitive(new ClassKeepTransitive(pattern));
    }

    @TaskAction
    public void run() throws Exception {
        FileCollection inputFiles = sourceFiles.filter(new FilterSpec("Excluding archive", toPatterns(archiveBypasses, archiveExcludes), false));
        final File outputFile = getDestinationPath();
        outputFile.getParentFile().mkdirs();
        getLogger().info("Running jarjar for {}", outputFile);
        getLogger().info("Inputs are {}", inputFiles);

        JarTransformer transformer = new JarTransformer(outputFile, processor);
        transformer.transform(new ClassPath(getProject().getProjectDir(), inputFiles));
    }
}
