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
package com.google.devtools.build.lib.rules.cpp;

/** Class holding constants for all C++ action names */
public class CppActionNames {

  /** A string constant used to compute CC_FLAGS make variable value */
  public static final String CC_FLAGS_MAKE_VARIABLE = "cc-flags-make-variable";
  /** A string constant for the strip action name. */
  public static final String STRIP = "strip";
  /** A string constant for the object copy action name. */
  public static final String OBJ_COPY = "objcopy_embed_data";
  /** A string constant for the linkstamp-compile action. */
  public static final String LINKSTAMP_COMPILE = "linkstamp-compile";
  /** A string constant for the c compilation action. */
  public static final String C_COMPILE = "c-compile";
  /** A string constant for the c++ compilation action. */
  public static final String CPP_COMPILE = "c++-compile";
  /** A string constant for the c++ module compile action. */
  public static final String CPP_MODULE_CODEGEN = "c++-module-codegen";
  /** A string constant for the objc compilation action. */
  public static final String OBJC_COMPILE = "objc-compile";
  /** A string constant for the objc++ compile action. */
  public static final String OBJCPP_COMPILE = "objc++-compile";
  /** A string constant for the c header parsing. */
  public static final String C_HEADER_PARSING = "c-header-parsing";
  /** A string constant for the c++ header parsing. */
  public static final String CPP_HEADER_PARSING = "c++-header-parsing";
  /**
   * A string constant for the c++ module compilation action. Note: currently we don't support C
   * module compilation.
   */
  public static final String CPP_MODULE_COMPILE = "c++-module-compile";
  /** A string constant for the assembler actions. */
  public static final String ASSEMBLE = "assemble";

  public static final String PREPROCESS_ASSEMBLE = "preprocess-assemble";
  /**
   * A string constant for the clif actions. Bazel enables different features of the toolchain based
   * on the name of the action. This name enables the clif_matcher feature, which switches the
   * "compiler" to the clif_matcher and adds some additional arguments as described in the CROSSTOOL
   * file.
   */
  public static final String CLIF_MATCH = "clif-match";

  /** Name of the action producing static library. */
  public static final String CPP_LINK_STATIC_LIBRARY = "c++-link-static-library";
  /** Name of the action producing dynamic library from cc_library. */
  public static final String CPP_LINK_NODEPS_DYNAMIC_LIBRARY = "c++-link-nodeps-dynamic-library";
  /** Name of the action producing dynamic library from cc_binary. */
  public static final String CPP_LINK_DYNAMIC_LIBRARY = "c++-link-dynamic-library";
  /** Name of the action producing executable binary. */
  public static final String CPP_LINK_EXECUTABLE = "c++-link-executable";
  /** Name of the objc action producing dynamic library */
  public static final String OBJC_FULLY_LINK = "objc-fully-link";
  /** Name of the objc action producing objc executable binary */
  public static final String OBJC_EXECUTABLE = "objc-executable";

  public static final String LTO_INDEXING = "lto-indexing";
  /** Name of the action producing thinlto index for dynamic library. */
  public static final String LTO_INDEX_DYNAMIC_LIBRARY = "lto-index-for-dynamic-library";
  /** Name of the action producing thinlto index for nodeps dynamic library. */
  public static final String LTO_INDEX_NODEPS_DYNAMIC_LIBRARY =
      "lto-index-for-nodeps-dynamic-library";
  /** Name of the action producing thinlto index for executable binary. */
  public static final String LTO_INDEX_EXECUTABLE = "lto-index-for-executable";

  public static final String LTO_BACKEND = "lto-backend";

  public static final String CPP_HEADER_ANALYSIS = "c++-header-analysis";
}
