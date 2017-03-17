// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.android.ide.common.res2.MergingException;
import com.android.utils.StdLogger;
import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.PathListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Generates the R class for an android_library with made up field initializers for the ids. The
 * real ids will be assigned when the android_binary is built.
 *
 * <p>Collects the R class fields from the parsed resource, and then writes out the resource class
 * files to a jar.
 */
public class LibraryRClassGeneratorAction {

  private static final Logger logger =
      Logger.getLogger(LibraryRClassGeneratorAction.class.getName());

  private static final StdLogger stdLogger = new StdLogger(StdLogger.Level.WARNING);

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {
    @Option(
      name = "classJarOutput",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      help = "Path for the generated java class jar."
    )
    public Path classJarOutput;

    @Option(
      name = "packageForR",
      defaultValue = "null",
      category = "config",
      help = "Custom java package to generate the R symbols files."
    )
    public String packageForR;

    @Option(
      name = "symbols",
      defaultValue = "",
      converter = PathListConverter.class,
      category = "config",
      help = "Parsed symbol binaries to write as R classes."
    )
    public List<Path> symbols;
  }

  public static void main(String[] args) throws Exception {
    final Stopwatch timer = Stopwatch.createStarted();
    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(Options.class, AaptConfigOptions.class);
    optionsParser.enableParamsFileSupport(FileSystems.getDefault());
    optionsParser.parseAndExitUponError(args);
    AaptConfigOptions aaptConfigOptions = optionsParser.getOptions(AaptConfigOptions.class);
    Options options = optionsParser.getOptions(Options.class);
    logger.fine(
        String.format("Option parsing finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    try (ScopedTemporaryDirectory scopedTmp =
        new ScopedTemporaryDirectory("android_resource_generated")) {
      AndroidResourceClassWriter resourceClassWriter =
          AndroidResourceClassWriter.createWith(
              aaptConfigOptions.androidJar,
              scopedTmp.getPath(),
              Strings.nullToEmpty(options.packageForR));
      resourceClassWriter.setIncludeClassFile(true);
      resourceClassWriter.setIncludeJavaFile(false);
      logger.fine(String.format("Setup finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      final ParsedAndroidData data =
          AndroidDataDeserializer.deserializeSymbolsToData(options.symbols);
      logger.fine(
          String.format("Deserialization finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      data.writeResourcesTo(resourceClassWriter);
      resourceClassWriter.flush();
      logger.fine(
          String.format("R writing finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      AndroidResourceOutputs.createClassJar(scopedTmp.getPath(), options.classJarOutput);
      logger.fine(
          String.format(
              "Creating class jar finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    } catch (IOException | MergingException | DeserializationException e) {
      logger.log(Level.SEVERE, "Errors during R generation.", e);
      throw e;
    }
  }
}
