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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCaseForJunit4;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for toolchain features.
 */
@RunWith(JUnit4.class)
public class CcToolchainTest extends BuildViewTestCaseForJunit4 {

  @Test
  public void testModuleMapAttribute() throws Exception {
    scratchConfiguredTarget("modules/map", "c",
        "cc_toolchain(",
        "    name = 'c',",
        "    module_map = 'map',",
        "    cpu = 'cherry',",
        "    compiler_files = 'compile-cherry',",
        "    dwp_files = 'dwp-cherry',",
        "    linker_files = 'link-cherry',",
        "    strip_files = ':every-file',",
        "    objcopy_files = 'objcopy-cherry',",
        "    all_files = ':every-file',",
        "    dynamic_runtime_libs = ['dynamic-runtime-libs-cherry'],",
        "    static_runtime_libs = ['static-runtime-libs-cherry'])");
  }
  
  @Test
  public void testModuleMapAttributeOptional() throws Exception {
    scratchConfiguredTarget("modules/map", "c",
        "cc_toolchain(",
        "    name = 'c',",
        "    cpu = 'cherry',",
        "    compiler_files = 'compile-cherry',",
        "    dwp_files = 'dwp-cherry',",
        "    linker_files = 'link-cherry',",
        "    strip_files = ':every-file',",
        "    objcopy_files = 'objcopy-cherry',",
        "    all_files = ':every-file',",
        "    dynamic_runtime_libs = ['dynamic-runtime-libs-cherry'],",
        "    static_runtime_libs = ['static-runtime-libs-cherry'])");    
  }

  @Test
  public void testFailWithMultipleModuleMaps() throws Exception {
    checkError("modules/multiple", "c", "expected a single artifact",
        "filegroup(name = 'multiple-maps', srcs = ['a.cppmap', 'b.cppmap'])",
        "cc_toolchain(",
        "    name = 'c',",
        "    module_map = ':multiple-maps',",
        "    cpu = 'cherry',",
        "    compiler_files = 'compile-cherry',",
        "    dwp_files = 'dwp-cherry',",
        "    linker_files = 'link-cherry',",
        "    strip_files = ':every-file',",
        "    objcopy_files = 'objcopy-cherry',",
        "    all_files = ':every-file',",
        "    dynamic_runtime_libs = ['dynamic-runtime-libs-cherry'],",
        "    static_runtime_libs = ['static-runtime-libs-cherry'])");
  }
}
