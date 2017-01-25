// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import com.android.dx.cf.direct.DirectClassFile;
import com.android.dx.cf.direct.StdAttributeFactory;
import com.android.dx.command.DxConsole;
import com.android.dx.dex.DexOptions;
import com.android.dx.dex.cf.CfOptions;
import com.android.dx.dex.cf.CfTranslator;
import com.android.dx.dex.code.PositionList;
import com.android.dx.dex.file.ClassDefItem;
import com.android.dx.dex.file.DexFile;
import com.android.dx.util.ByteArray;
import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

/**
 * Common helper class that encodes Java classes into {@link DexFile}s.
 */
class Dexing {

  /**
   * Common command line options for use with {@link Dexing}.
   */
  public static class DexingOptions extends OptionsBase {

    @Option(name = "locals",
        defaultValue = "true", // dx's default
        category = "semantics",
        allowMultiple = false,
        help = "Whether to include local variable tables (useful for debugging).")
    public boolean localInfo;

    @Option(name = "optimize",
        defaultValue = "true", // dx's default
        category = "semantics",
        allowMultiple = false,
        help = "Whether to do SSA/register optimization.")
    public boolean optimize;

    @Option(name = "warning",
        defaultValue = "true", // dx's default
        category = "misc",
        allowMultiple = false,
        help = "Whether to print warnings.")
    public boolean printWarnings;

    public CfOptions toCfOptions() {
      CfOptions result = new CfOptions();
      result.localInfo = this.localInfo;
      result.optimize = this.optimize;
      result.warn = printWarnings ? DxConsole.err : DxConsole.noop;
      // Use dx's defaults
      result.optimizeListFile = null;
      result.dontOptimizeListFile = null;
      result.positionInfo = PositionList.LINES;
      result.strictNameCheck = true;
      result.statistics = false; // we're not supporting statistics anyways
      return result;
    }

    public DexOptions toDexOptions() {
      DexOptions result = new DexOptions();
      result.forceJumbo = false; // dx's default
      return result;
    }
  }

  /**
   * Class file and possible dexing options, to look up dexing results in caches.
   */
  @AutoValue
  abstract static class DexingKey {
    static DexingKey create(boolean localInfo, boolean optimize, byte[] classfileContent) {
      // TODO(bazel-team): Maybe we can use a minimal collision hash instead of full content
      return new AutoValue_Dexing_DexingKey(localInfo, optimize, classfileContent);
    }

    /** Returns whether {@link CfOptions#localInfo local variable information} is included. */
    abstract boolean localInfo();

    /** Returns whether {@link CfOptions#optimize SSA/register optimization} is performed. */
    abstract boolean optimize();

    /** Returns the class file to dex, <b>not</b> the dexed class. Don't modify the return value! */
    @SuppressWarnings("mutable") abstract byte[] classfileContent();
  }

  private final DexOptions dexOptions;
  private final CfOptions cfOptions;

  public Dexing(DexingOptions options) {
    this(options.toDexOptions(), options.toCfOptions());
  }

  @VisibleForTesting
  Dexing(DexOptions dexOptions, CfOptions cfOptions) {
    this.dexOptions = dexOptions;
    this.cfOptions = cfOptions;
  }

  public static DirectClassFile parseClassFile(byte[] classfile, String classfilePath) {
    DirectClassFile result = new DirectClassFile(
        new ByteArray(classfile), classfilePath, /*strictParse*/ false);
    result.setAttributeFactory(StdAttributeFactory.THE_ONE);
    result.getMagic(); // triggers the parsing
    return result;
  }

  public DexFile newDexFile() {
    return new DexFile(dexOptions);
  }

  public ClassDefItem addToDexFile(DexFile dest, DirectClassFile classFile) {
    ClassDefItem result = CfTranslator.translate(classFile,
        (byte[]) null /*ignored*/,
        cfOptions,
        dest.getDexOptions(),
        dest);
    dest.add(result);
    return result;
  }

  public DexingKey getDexingKey(byte[] classfile) {
    return DexingKey.create(cfOptions.localInfo, cfOptions.optimize, classfile);
  }
}
