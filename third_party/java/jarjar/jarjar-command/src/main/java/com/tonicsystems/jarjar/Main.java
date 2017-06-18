/**
 * Copyright 2007 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.tonicsystems.jarjar;

import com.tonicsystems.jarjar.classpath.ClassPath;
import com.tonicsystems.jarjar.transform.jar.DefaultJarProcessor;
import com.tonicsystems.jarjar.transform.config.RulesFileParser;
import com.tonicsystems.jarjar.dependencies.TextDependencyHandler;
import com.tonicsystems.jarjar.dependencies.DependencyFinder;
import com.tonicsystems.jarjar.dependencies.DependencyHandler;
import com.tonicsystems.jarjar.strings.StringDumper;
import com.tonicsystems.jarjar.transform.config.AbstractPattern;
import com.tonicsystems.jarjar.transform.JarTransformer;
import java.io.File;
import java.io.IOException;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;
import joptsimple.OptionParser;
import joptsimple.OptionSet;
import joptsimple.OptionSpec;

public class Main {

    private static final String LINE_SEPARATOR = System.getProperty("line.separator");
    private static final String PATH_SEPARATOR = System.getProperty("path.separator");

    public static enum Mode {

        strings, find, process;
    }

    private final OptionParser parser = new OptionParser();
    private final OptionSpec<Void> helpOption = parser.accepts("help")
            .forHelp();
    private final OptionSpec<Mode> modeOption = parser.accepts("mode")
            .withRequiredArg().ofType(Mode.class).defaultsTo(Mode.process).describedAs("Mode to run (strings, find, process)");
    private final OptionSpec<DependencyHandler.Level> levelOption = parser.accepts("level")
            .withRequiredArg().ofType(DependencyHandler.Level.class).defaultsTo(DependencyHandler.Level.CLASS).describedAs("Level for DepHandler.");
    private final OptionSpec<File> fromFilesOption = parser.accepts("from")
            .withRequiredArg().ofType(File.class).withValuesSeparatedBy(PATH_SEPARATOR).describedAs("Classpath for strings, find.");
    private final OptionSpec<File> rulesOption = parser.accepts("rules")
            .withRequiredArg().ofType(File.class).describedAs("Rules file.");
    private final OptionSpec<File> outputOption = parser.accepts("output")
            .withRequiredArg().ofType(File.class).describedAs("Output JAR file.");
    private final OptionSpec<File> filesOption = parser.nonOptions()
            .ofType(File.class).describedAs("JAR files or directories to process.");

    public void run(@Nonnull String[] args) throws Exception {
        OptionSet options = parser.parse(args);
        if (options.has(helpOption)) {
            parser.printHelpOn(System.err);
            System.exit(1);
        }

        Mode mode = options.valueOf(modeOption);
        switch (mode) {
            case find:
                find(options);
                break;
            case process:
                process(options);
                break;
            case strings:
                strings(options);
                break;
            default:
                throw new IllegalArgumentException("Illegal mode " + mode);
        }
    }

    private static boolean isEmpty(@CheckForNull List<?> values) {
        if (values == null)
            return true;
        return values.isEmpty();
    }

    @Nonnull
    private <T> T valueOf(@Nonnull OptionSet options, @Nonnull OptionSpec<T> option) {
        T value = options.valueOf(option);
        if (value == null)
            throw new IllegalArgumentException(option + " is required.");
        return value;
    }

    @Nonnull
    private <T> List<T> valuesOf(@Nonnull OptionSet options, @Nonnull OptionSpec<T> option) {
        List<T> values = options.valuesOf(option);
        if (isEmpty(values))
            throw new IllegalArgumentException(option + " is required.");
        return values;
    }

    private static ClassPath newClassPath(Iterable<? extends File> files) {
        return new ClassPath(new File(System.getProperty("user.dir")), files);
    }

    public void strings(@Nonnull OptionSet options) throws IOException {
        List<File> files = options.valuesOf(filesOption);
        File parent = new File(System.getProperty("user.dir"));
        new StringDumper().run(System.out, newClassPath(files));
        System.out.flush();
    }

    public void find(@Nonnull OptionSet options) throws IOException {
        List<File> toFiles = valuesOf(options, filesOption);
        List<File> fromFiles = options.valuesOf(fromFilesOption);
        if (isEmpty(fromFiles))
            fromFiles = toFiles;
        DependencyHandler.Level level = valueOf(options, levelOption);
        DependencyHandler handler = new TextDependencyHandler(System.out, level);
        new DependencyFinder().run(handler, newClassPath(fromFiles), newClassPath(toFiles));
        System.out.flush();
    }

    public void process(@Nonnull OptionSet options) throws IOException {
        File outputFile = valueOf(options, outputOption);
        File rulesFile = valueOf(options, rulesOption);
        List<File> files = valuesOf(options, filesOption);

        DefaultJarProcessor processor = new DefaultJarProcessor();
        RulesFileParser.parse(processor, rulesFile);
        processor.setSkipManifest(Boolean.getBoolean("skipManifest"));

        JarTransformer transformer = new JarTransformer(outputFile, processor);
        transformer.transform(newClassPath(files));
    }

    public static void main(String[] args) throws Exception {
        Main main = new Main();
        main.run(args);
        // MainUtil.runMain(new Main(), args, "help");
    }
}
