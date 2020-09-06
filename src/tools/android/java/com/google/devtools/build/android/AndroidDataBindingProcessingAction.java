// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.android.builder.core.VariantType;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.VariantTypeConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Action that processes resources for Databinding v2.
 */
public class AndroidDataBindingProcessingAction {

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {

    @Option(
        name = "resource_root",
        defaultValue = "null",
        converter =  PathConverter.class,
        allowMultiple =  true,
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Input resource root. A corresponding output resource root must be passed to "
            + "--output_resource_root. Multiple roots will be paired in the order they're passed.")
    public List<Path> resourceRoots;

    @Option(
        name = "output_resource_directory",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "The output resource directory. Input source roots will be appended to this to "
                + "create the output resource roots.")
    public Path outputResourceDirectory;

    @Option(
        name = "packageType",
        defaultValue = "DEFAULT",
        converter = VariantTypeConverter.class,
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Variant configuration type for packaging the resources."
                + " Acceptable values DEFAULT, LIBRARY, ANDROID_TEST, UNIT_TEST")
    public VariantType packageType;

    @Option(
        name = "dataBindingInfoOut",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to where data binding's layout info output should be written.")
    public Path dataBindingInfoOut;

    @Option(
        name = "appId",
        defaultValue = "null",
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "The java package for the app.")
    public String appId;
  }

  public static void main(String[] args) throws IOException {

    OptionsParser optionsParser =
        OptionsParser.builder()
            .allowResidue(true)
            .optionsClasses(
                Options.class, AaptConfigOptions.class, ResourceProcessorCommonOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);
    Options options = optionsParser.getOptions(Options.class);

    if (options.dataBindingInfoOut == null) {
      throw new IllegalArgumentException("--dataBindingInfoOut is required");
    }

    if (options.appId == null) {
      throw new IllegalArgumentException("--appId is required");
    }

    if (options.outputResourceDirectory == null) {
      throw new IllegalArgumentException("--output_resource_directory is required");
    }

    List<Path> resourceRoots = options.resourceRoots == null
        ? Collections.emptyList()
        : options.resourceRoots;

    try (ScopedTemporaryDirectory dataBindingInfoOutDir =
        new ScopedTemporaryDirectory("android_data_binding_layout_info_tmp")) {

      // 1. Process databinding resources for each source root.
      for (Path resourceRoot : resourceRoots) {

        AndroidResourceProcessor.processDataBindings(
            options.outputResourceDirectory,
            resourceRoot,
            dataBindingInfoOutDir.getPath(),
            options.appId,
            /* shouldZipDataBindingInfo= */ false);
      }

      // 2. Zip all the layout info files into one zip file.
      try (ZipOutputStream layoutInfoZip =
              new ZipOutputStream(Files.newOutputStream(options.dataBindingInfoOut));
          Stream<Path> layoutInfos = Files.list(dataBindingInfoOutDir.getPath())) {
        Iterator<Path> it = layoutInfos.iterator();
        while (it.hasNext()) {
          Path layoutInfo = it.next();
          ZipEntry zipEntry = new ZipEntry(layoutInfo.getFileName().toString());
          layoutInfoZip.putNextEntry(zipEntry);
          Files.copy(layoutInfo, layoutInfoZip);
          layoutInfoZip.closeEntry();
        }
      }
    }
  }
}
