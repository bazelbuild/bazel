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
package com.google.devtools.build.lib.packages.util;

import com.google.common.base.Predicate;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.protobuf.TextFormat;

import java.io.IOException;

/**
 * Creates mock BUILD files required for the C/C++ rules.
 */
public abstract class MockCcSupport {

  /**
   * Filter to remove implicit crosstool artifact and module map inputs
   * of C/C++ rules.
   */
  public static final Predicate<Artifact> CC_ARTIFACT_FILTER =
      new Predicate<Artifact>() {
        @Override
        public boolean apply(Artifact artifact) {
          String basename = artifact.getExecPath().getBaseName();
          String pathString = artifact.getExecPathString();
          return !pathString.startsWith("third_party/crosstool/")
              && !(pathString.contains("/internal/_middlemen") && basename.contains("crosstool"))
              && !pathString.startsWith("_bin/build_interface_so")
              && !pathString.endsWith(".cppmap");
        }
      };

  /**
   * A feature configuration snippet useful for testing header processing.
   */
  public static final String HEADER_PROCESSING_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'parse_headers'"
          + "  flag_set {"
          + "    action: 'c++-header-parsing'"
          + "    flag_group {"
          + "      flag: '<c++-header-parsing>'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'preprocess_headers'"
          + "  flag_set {"
          + "    action: 'c++-header-preprocessing'"
          + "    flag_group {"
          + "      flag: '<c++-header-preprocessing>'"
          + "    }"
          + "  }"
          + "}";

  /**
   * A feature configuration snippet useful for testing the layering check.
   */
  public static final String LAYERING_CHECK_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'layering_check'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-header-preprocessing'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      flag: 'dependent_module_map_file:%{dependent_module_map_files}'"
          + "    }"
          + "  }"
          + "}";

  /**
   * A feature configuration snippet useful for testing header modules.
   */
  public static final String HEADER_MODULES_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'header_modules'"
          + "  implies: 'module_maps'"
          + "  implies: 'use_header_modules'"
          + "}"
          + "feature {"
          + "  name: 'module_maps'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-header-preprocessing'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      flag: 'module_name:%{module_name}'"
          + "      flag: 'module_map_file:%{module_map_file}'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'use_header_modules'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-header-preprocessing'"
          + "    action: 'c++-modules-compile'"
          + "    flag_group {"
          + "      flag: 'module_file:%{module_files}'"
          + "    }"
          + "  }"
          + "}";

  /**
   * A feature configuration snippet useful for testing environment variables.
   */
  public static final String ENV_VAR_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'env_feature'"
          + "  implies: 'static_env_feature'"
          + "  implies: 'module_maps'"
          + "}"
          + "feature {"
          + "  name: 'static_env_feature'"
          + "  env_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-header-preprocessing'"
          + "    action: 'c++-module-compile'"
          + "    env_entry {"
          + "      key: 'cat'"
          + "      value: 'meow'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'module_maps'"
          + "  env_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-header-preprocessing'"
          + "    action: 'c++-module-compile'"
          + "    env_entry {"
          + "      key: 'module'"
          + "      value: 'module_name:%{module_name}'"
          + "    }"
          + "  }"
          + "}";

  public static final String THIN_LTO_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'thin_lto'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    flag_group {"
          + "      flag: '-Xclang-only=-Wno-inconsistent-missing-override'"
          + "      flag: '-flto'"
          + "      flag: '-O2'"
          + "    }"
          + "  }"
          + "}";

  /** Filter to remove implicit dependencies of C/C++ rules. */
  private final Predicate<Label> ccLabelFilter =
      new Predicate<Label>() {
        @Override
        public boolean apply(Label label) {
          return labelNameFilter().apply("//" + label.getPackageName());
        }
      };

  public static String mergeCrosstoolConfig(String original, CToolchain toolchain)
      throws TextFormat.ParseException {
    CrosstoolConfig.CrosstoolRelease.Builder builder =
        CrosstoolConfig.CrosstoolRelease.newBuilder();
    TextFormat.merge(original, builder);
    for (CToolchain.Builder toolchainBuilder : builder.getToolchainBuilderList()) {
      toolchainBuilder.mergeFrom(toolchain);
    }
    return TextFormat.printToString(builder.build());
  }

  public abstract Predicate<String> labelNameFilter();

  /**
   * Setup the support for building C/C++.
   */
  public abstract void setup(MockToolsConfig config) throws IOException;

  public void setupCrosstoolWithEmbeddedRuntimes(MockToolsConfig config) throws IOException {
    createCrosstoolPackage(config, true);
  }

  /**
   * Creates a crosstool package by merging {@code toolchain} with the default mock CROSSTOOL file.
   *
   * @param partialToolchain A string representation of a CToolchain protocol buffer; note that
   *        this is allowed to be a partial buffer (required fields may be omitted).
   */
  public void setupCrosstool(MockToolsConfig config, String partialToolchain) throws IOException {
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    TextFormat.merge(partialToolchain, toolchainBuilder);
    setupCrosstool(config, toolchainBuilder.buildPartial());
  }

  /**
   * Creates a crosstool package by merging {@code toolchain} with the default mock CROSSTOOL file.
   */
  public void setupCrosstool(MockToolsConfig config, CToolchain toolchain) throws IOException {
    createCrosstoolPackage(
        config, /* addEmbeddedRuntimes= */ false, /* addModuleMap= */ true, null, null, toolchain);
  }

  /**
   * Create a crosstool package. For integration tests, it actually links in a working crosstool,
   * for all other tests, it only creates a dummy package, with a working CROSSTOOL file. The code
   * here matches the declarations in {@link CrosstoolTestHelper}.
   *
   * <p>If <code>addEmbeddedRuntimes</code> is true, it also adds filegroups for the embedded
   * runtimes.
   */
  public void setupCrosstool(
      MockToolsConfig config,
      boolean addEmbeddedRuntimes,
      boolean addModuleMap,
      String staticRuntimesLabel,
      String dynamicRuntimesLabel,
      CToolchain toolchain)
      throws IOException {
    createCrosstoolPackage(
        config,
        addEmbeddedRuntimes,
        addModuleMap,
        staticRuntimesLabel,
        dynamicRuntimesLabel,
        toolchain);
  }

  protected static void createToolsCppPackage(MockToolsConfig config) throws IOException {
    config.create(
        "tools/cpp/BUILD",
        "cc_library(name = 'stl')",
        "filegroup(name='toolchain', srcs=['//third_party/crosstool'])");
  }

  protected void createCrosstoolPackage(MockToolsConfig config, boolean addEmbeddedRuntimes)
      throws IOException {
    createCrosstoolPackage(config, addEmbeddedRuntimes, /*addModuleMap=*/ true, null, null, null);
  }

  protected String getCrosstoolTopPathForConfig(MockToolsConfig config) {
    if (config.isRealFileSystem()) {
      return getRealFilesystemCrosstoolTopPath();
    } else {
      return getMockCrosstoolPath();
    }
  }

  public abstract String getMockCrosstoolPath();

  public static PackageIdentifier getMockCrosstoolsTop() {
    try {
      return PackageIdentifier.create(
          RepositoryName.create(TestConstants.TOOLS_REPOSITORY),
          new PathFragment(TestConstants.TOOLS_REPOSITORY_PATH));
    } catch (LabelSyntaxException e) {
      Verify.verify(false);
      throw new AssertionError();
    }
  }

  protected void createCrosstoolPackage(
      MockToolsConfig config,
      boolean addEmbeddedRuntimes,
      boolean addModuleMap,
      String staticRuntimesLabel,
      String dynamicRuntimesLabel,
      CToolchain toolchain)
      throws IOException {
    String crosstoolTop = getCrosstoolTopPathForConfig(config);
    if (config.isRealFileSystem()) {
      config.linkTools(getRealFilesystemTools(crosstoolTop));
    } else {
      String crosstoolFile = readCrosstoolFile();
      if (toolchain != null) {
        crosstoolFile = mergeCrosstoolConfig(crosstoolFile, toolchain);
      }
      new Crosstool(config, crosstoolTop)
          .setEmbeddedRuntimes(addEmbeddedRuntimes, staticRuntimesLabel, dynamicRuntimesLabel)
          .setCrosstoolFile(getMockCrosstoolVersion(), crosstoolFile)
          .setSupportedArchs(getCrosstoolArchs())
          .setAddModuleMap(addModuleMap)
          .setSupportsHeaderParsing(true)
          .write();
    }
  }

  protected abstract String getMockCrosstoolVersion();

  protected abstract String readCrosstoolFile() throws IOException;

  protected abstract ImmutableList<String> getCrosstoolArchs();

  protected abstract String[] getRealFilesystemTools(String crosstoolTop);

  protected abstract String getRealFilesystemCrosstoolTopPath();

  public final Predicate<Label> labelFilter() {
    return ccLabelFilter;
  }
}
