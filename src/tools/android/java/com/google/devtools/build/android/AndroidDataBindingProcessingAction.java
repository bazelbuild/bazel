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

import com.android.builder.core.VariantTypeImpl;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.Parameters;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import com.google.devtools.build.android.Converters.CompatVariantTypeConverter;
import java.io.IOException;
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
  @Parameters(separators = "= ")
  public static final class Options extends OptionsBaseWithResidue {

    @Parameter(
        names = "--resource_root",
        converter = CompatPathConverter.class,
        description =
            "Input resource root. A corresponding output resource root must be passed to"
                + " --output_resource_root. Multiple roots will be paired in the order they're"
                + " passed.")
    public List<Path> resourceRoots = ImmutableList.of();

    @Parameter(
        names = "--output_resource_directory",
        converter = CompatPathConverter.class,
        description =
            "The output resource directory. Input source roots will be appended to this to "
                + "create the output resource roots.")
    public Path outputResourceDirectory;

    @Parameter(
        names = "--packageType",
        converter = CompatVariantTypeConverter.class,
        description =
            "Variant configuration type for packaging the resources."
                + " Acceptable values BASE_APK, LIBRARY, ANDROID_TEST, UNIT_TEST")
    public VariantTypeImpl packageType = VariantTypeImpl.BASE_APK;

    @Parameter(
        names = "--dataBindingInfoOut",
        converter = CompatPathConverter.class,
        description = "Path to where data binding's layout info output should be written.")
    public Path dataBindingInfoOut;

    @Parameter(names = "--appId", description = "The java package for the app.")
    public String appId;
  }

  public static void main(String[] args) throws ParameterException, IOException {
    Options options = new Options();
    AaptConfigOptions aaptConfigOptions = new AaptConfigOptions();
    ResourceProcessorCommonOptions resourceProcessorCommonOptions =
        new ResourceProcessorCommonOptions();
    Object[] allOptions = new Object[] {options, aaptConfigOptions, resourceProcessorCommonOptions};
    JCommander jc = new JCommander(allOptions);
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(allOptions, preprocessedArgs);
    jc.parse(normalizedArgs);

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
            /* shouldZipDataBindingInfo= */ false,
            aaptConfigOptions.useDataBindingAndroidX);
      }

      // 2. Zip all the layout info files into one zip file.
      try (ZipOutputStream layoutInfoZip =
              new ZipOutputStream(Files.newOutputStream(options.dataBindingInfoOut));
          Stream<Path> layoutInfos = Files.list(dataBindingInfoOutDir.getPath())) {
        Iterator<Path> it = layoutInfos.iterator();
        while (it.hasNext()) {
          Path layoutInfo = it.next();
          ZipEntry zipEntry = new ZipEntry(layoutInfo.getFileName().toString());
          zipEntry.setTime(0); // for deterministic output
          layoutInfoZip.putNextEntry(zipEntry);
          Files.copy(layoutInfo, layoutInfoZip);
          layoutInfoZip.closeEntry();
        }
      }
    }
  }
}
