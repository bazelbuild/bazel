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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain.Builder;
import java.util.List;

/**
 * Class that contains implementations of NdkStlImpl, one for each STL implementation of the STL
 * in the Android NDK.
 */
public final class StlImpls {

  public static final String DEFAULT_STL_NAME = "gnu-libstdcpp";
  
  private StlImpls() {}
  
  public static class GnuLibStdCppStlImpl extends StlImpl {

    public GnuLibStdCppStlImpl(NdkPaths ndkPaths) {
      super(DEFAULT_STL_NAME, ndkPaths);
    }

    @Override
    public void addStlImpl(Builder toolchain, String gccVersion) {
      addBaseStlImpl(toolchain, gccVersion);
      toolchain.addAllUnfilteredCxxFlag(createIncludeFlags(
          ndkPaths.createGnuLibstdcIncludePaths(gccVersion, toolchain.getTargetCpu())));
    }
  }

  public static class LibCppStlImpl extends StlImpl {

    public LibCppStlImpl(NdkPaths ndkPaths) {
      super("libcpp", ndkPaths);
    }

    @Override
    public void addStlImpl(Builder toolchain, String gccVersion) {
      addBaseStlImpl(toolchain, null);
      toolchain.addAllUnfilteredCxxFlag(createIncludeFlags(ndkPaths.createLibcxxIncludePaths()));
    }
  }

  public static class StlPortStlImpl extends StlImpl {

    public StlPortStlImpl(NdkPaths ndkPaths) {
      super("stlport", ndkPaths);
    }

    @Override
    public void addStlImpl(Builder toolchain, String gccVersion) {
      addBaseStlImpl(toolchain, null);
      toolchain.addAllUnfilteredCxxFlag(createIncludeFlags(ndkPaths.createStlportIncludePaths()));
    }
  }

  /**
   * @param ndkPaths NdkPaths to use for creating the NdkStlImpls
   * @return an ImmutableList of every available NdkStlImpl
   */
  public static List<StlImpl> get(NdkPaths ndkPaths) {
    return ImmutableList.of(
        new GnuLibStdCppStlImpl(ndkPaths),
        new StlPortStlImpl(ndkPaths),
        new LibCppStlImpl(ndkPaths));
  }
}
