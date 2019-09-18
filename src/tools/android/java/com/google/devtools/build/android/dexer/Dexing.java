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
import com.android.dx.command.dexer.DxContext;
import com.android.dx.dex.DexOptions;
import com.android.dx.dex.cf.CfOptions;
import com.android.dx.dex.cf.CfTranslator;
import com.android.dx.dex.code.PositionList;
import com.android.dx.dex.file.ClassDefItem;
import com.android.dx.dex.file.DexFile;
import com.android.dx.util.ByteArray;
import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.io.ByteStreams;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.PrintStream;
import java.lang.reflect.Field;

/**
 * Common helper class that encodes Java classes into {@link DexFile}s.
 */
class Dexing {

  static final PrintStream nullout = new PrintStream(ByteStreams.nullOutputStream());

  /**
   * Parser for positions options based on the integer field names in {@link PositionList}.
   */
  public static class PositionGranularityConverter implements Converter<Integer> {
    @Override
    public Integer convert(String input) throws OptionsParsingException {
      for (Field field : PositionList.class.getFields()) {
        if (field.getName().equalsIgnoreCase(input)) {
          try {
            return field.getInt(null);
          } catch (RuntimeException | IllegalAccessException e) {
            throw new OptionsParsingException("Can't parse positions option", input, e);
          }
        }
      }
      throw new OptionsParsingException("Unknown positions option", input);
    }

    @Override
    public String getTypeDescription() {
      return "One of the options from dx's --positions flag";
    }
  }

  /**
   * Common command line options for use with {@link Dexing}.
   */
  public static class DexingOptions extends OptionsBase {

    @Option(
      name = "locals",
      defaultValue = "true", // dx's default
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = false,
      help = "Whether to include local variable tables (useful for debugging)."
    )
    public boolean localInfo;

    @Option(
      name = "optimize",
      defaultValue = "true", // dx's default
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = false,
      help = "Whether to do SSA/register optimization."
    )
    public boolean optimize;

    @Option(
      name = "positions",
      defaultValue = "lines", // dx's default
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = false,
      converter = PositionGranularityConverter.class,
      help = "How densely to emit line number information."
    )
    // Note this field must be initialized (usually done by an options parser) since the implicit
    // initial value 0 is not valid.
    public int positionInfo;

    @Option(
      name = "warning",
      defaultValue = "true", // dx's default
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = false,
      help = "Whether to print warnings."
    )
    public boolean printWarnings;

    @Option(
        name = "min_sdk_version",
        defaultValue = "13", // dx's default is DexFormat.API_NO_EXTENDED_OPCODES = 13
        category = "misc",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Min sdk version.")
    public int minSdkVersion;

    public CfOptions toCfOptions(DxContext context) {
      CfOptions result = new CfOptions();
      result.localInfo = this.localInfo;
      result.optimize = this.optimize;
      result.warn = printWarnings ? context.err : Dexing.nullout;
      // Use dx's defaults
      result.optimizeListFile = null;
      result.dontOptimizeListFile = null;
      result.positionInfo = positionInfo;
      result.strictNameCheck = true;
      result.statistics = false; // we're not supporting statistics anyways
      return result;
    }

    public DexOptions toDexOptions() {
      DexOptions result = new DexOptions();
      result.forceJumbo = false; // dx's default
      result.minSdkVersion = minSdkVersion;
      return result;
    }
  }

  /**
   * Class file and possible dexing options, to look up dexing results in caches.
   */
  @AutoValue
  abstract static class DexingKey {

    static DexingKey create(
        boolean localInfo,
        boolean optimize,
        int positionInfo,
        int minSdkVersion,
        byte[] classfileContent) {

      // TODO(bazel-team): Maybe we can use a minimal collision hash instead of full content
      return new AutoValue_Dexing_DexingKey(
          localInfo, optimize, positionInfo, minSdkVersion, classfileContent);
    }

    /** Returns whether {@link CfOptions#localInfo local variable information} is included. */
    abstract boolean localInfo();

    /** Returns whether {@link CfOptions#optimize SSA/register optimization} is performed. */
    abstract boolean optimize();

    /** Returns how much line number information is emitted as a {@link PositionList} constant. */
    abstract int positionInfo();

    /** Returns the min sdk version that the dex code was created for. */
    abstract int minSdkVersion();

    /** Returns the class file to dex, <b>not</b> the dexed class. Don't modify the return value! */
    @SuppressWarnings("mutable") abstract byte[] classfileContent();
  }

  private final DxContext context;
  private final DexOptions dexOptions;
  private final CfOptions cfOptions;

  public Dexing(DexingOptions options) {
    this(new DxContext(), options);
  }

  public Dexing(DxContext context, DexingOptions options) {
    this(context, options.toDexOptions(), options.toCfOptions(context));
  }

  @VisibleForTesting
  Dexing(DxContext context, DexOptions dexOptions, CfOptions cfOptions) {
    this.context = context;
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
    ClassDefItem result = CfTranslator.translate(
        context,
        classFile,
        (byte[]) null /*ignored*/,
        cfOptions,
        dest.getDexOptions(),
        dest);
    dest.add(result);
    return result;
  }

  public DexingKey getDexingKey(byte[] classfile) {
    return DexingKey.create(
        cfOptions.localInfo,
        cfOptions.optimize,
        cfOptions.positionInfo,
        dexOptions.minSdkVersion,
        classfile);
  }
}
