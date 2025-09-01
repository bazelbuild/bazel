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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.CollidingProvidesException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import java.util.List;
import java.util.regex.Pattern;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;

/** Common parts of the implementation of cc rules. */
public final class CcCommon implements StarlarkValue {

  /** Name of the build variable for the sysroot path variable name. */
  public static final String SYSROOT_VARIABLE_NAME = "sysroot";


  /** Name of the build variable for the minimum_os_version being targeted. */
  public static final String MINIMUM_OS_VERSION_VARIABLE_NAME = "minimum_os_version";

  public static final String PIC_CONFIGURATION_ERROR =
      "PIC compilation is requested but the toolchain does not support it "
          + "(feature named 'supports_pic' is not enabled)";

  public static final ImmutableSet<String> ALL_COMPILE_ACTIONS =
      ImmutableSet.of(
          CppActionNames.C_COMPILE,
          CppActionNames.CPP_COMPILE,
          CppActionNames.CPP_HEADER_PARSING,
          CppActionNames.CPP_MODULE_COMPILE,
          CppActionNames.CPP_MODULE_CODEGEN,
          CppActionNames.CPP_MODULE_DEPS_SCANNING,
          CppActionNames.CPP20_MODULE_COMPILE,
          CppActionNames.CPP20_MODULE_CODEGEN,
          CppActionNames.ASSEMBLE,
          CppActionNames.PREPROCESS_ASSEMBLE,
          CppActionNames.CLIF_MATCH,
          CppActionNames.LINKSTAMP_COMPILE,
          CppActionNames.CC_FLAGS_MAKE_VARIABLE,
          CppActionNames.LTO_BACKEND,
          CppActionNames.CPP_HEADER_ANALYSIS);

  public static final ImmutableSet<String> ALL_LINK_ACTIONS =
      ImmutableSet.of(
          CppActionNames.LTO_INDEX_EXECUTABLE,
          CppActionNames.LTO_INDEX_DYNAMIC_LIBRARY,
          CppActionNames.LTO_INDEX_NODEPS_DYNAMIC_LIBRARY,
          LinkTargetType.EXECUTABLE.getActionName(),
          Link.LinkTargetType.DYNAMIC_LIBRARY.getActionName(),
          Link.LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName());

  public static final ImmutableSet<String> ALL_ARCHIVE_ACTIONS =
      ImmutableSet.of(Link.LinkTargetType.STATIC_LIBRARY.getActionName());

  public static final ImmutableSet<String> ALL_OTHER_ACTIONS =
      ImmutableSet.of(CppActionNames.STRIP);

  /** Action configs we request to enable. */
  public static final ImmutableSet<String> DEFAULT_ACTION_CONFIGS =
      ImmutableSet.<String>builder()
          .addAll(ALL_COMPILE_ACTIONS)
          .addAll(ALL_LINK_ACTIONS)
          .addAll(ALL_ARCHIVE_ACTIONS)
          .addAll(ALL_OTHER_ACTIONS)
          .build();

  public static final ImmutableSet<String> OBJC_ACTIONS =
      ImmutableSet.of(
          CppActionNames.OBJC_COMPILE,
          CppActionNames.OBJCPP_COMPILE,
          CppActionNames.OBJC_FULLY_LINK,
          CppActionNames.OBJC_EXECUTABLE);

  /** An enum for the list of supported languages. */
  public enum Language {
    CPP("c++"),
    OBJC("objc"),
    OBJCPP("objc++");

    private final String representation;

    Language(String representation) {
      this.representation = representation;
    }

    public String getRepresentation() {
      return representation;
    }
  }

  /** A filter that removes copts from a c++ compile action according to a nocopts regex. */
  public static final class CoptsFilter implements StarlarkValue {
    private final Pattern noCoptsPattern;
    private final boolean allPasses;

    private CoptsFilter(Pattern noCoptsPattern, boolean allPasses) {
      this.noCoptsPattern = noCoptsPattern;
      this.allPasses = allPasses;
    }

    /** Creates a filter that filters all matches to a regex. */
    public static CoptsFilter fromRegex(Pattern noCoptsPattern) {
      return new CoptsFilter(noCoptsPattern, false);
    }

    /** Creates a filter that passes on all inputs. */
    public static CoptsFilter alwaysPasses() {
      return new CoptsFilter(null, true);
    }

    /**
     * Returns true if the provided string passes through the filter, or false if it should be
     * removed.
     */
    public boolean passesFilter(String flag) {
      if (allPasses) {
        return true;
      } else {
        return !noCoptsPattern.matcher(flag).matches();
      }
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Override
    public void repr(Printer printer) {
      printer.append("CoptsFilter(noCoptsPattern=");
      printer.append(noCoptsPattern == null ? "null" : noCoptsPattern.pattern());
      printer.append(", allPasses=");
      printer.append(Boolean.toString(allPasses));
      printer.append(")");
    }
  }

  @StarlarkMethod(name = "loose_include_dirs", structField = true, documented = false)
  public Sequence<String> getLooseIncludeDirsForStarlark() {
    return StarlarkList.empty();
  }

  public static ImmutableList<String> getCoverageFeatures(CppConfiguration cppConfiguration) {
    ImmutableList.Builder<String> coverageFeatures = ImmutableList.builder();
    if (cppConfiguration.collectCodeCoverage()) {
      coverageFeatures.add(CppRuleClasses.COVERAGE);
      if (cppConfiguration.useLLVMCoverageMapFormat()) {
        coverageFeatures.add(CppRuleClasses.LLVM_COVERAGE_MAP_FORMAT);
      } else {
        coverageFeatures.add(CppRuleClasses.GCC_COVERAGE_MAP_FORMAT);
      }
    }
    return coverageFeatures.build();
  }

  public static FeatureConfiguration configureFeaturesOrThrowEvalException(
      ImmutableSet<String> requestedFeatures,
      ImmutableSet<String> unsupportedFeatures,
      Language language,
      CcToolchainProvider toolchain,
      CppConfiguration cppConfiguration)
      throws EvalException {
    ImmutableSet.Builder<String> allRequestedFeaturesBuilder = ImmutableSet.builder();
    ImmutableSet.Builder<String> unsupportedFeaturesBuilder = ImmutableSet.builder();
    unsupportedFeaturesBuilder.addAll(unsupportedFeatures);
    if (!toolchain.supportsHeaderParsing()) {
      // TODO(b/159096411): Remove once supports_header_parsing has been removed from the
      // cc_toolchain rule.
      unsupportedFeaturesBuilder.add(CppRuleClasses.PARSE_HEADERS);
    }

    if (language != Language.OBJC
        && language != Language.OBJCPP
        && toolchain.getCcInfoCcCompilationContext().getCppModuleMap() == null) {
      unsupportedFeaturesBuilder.add(CppRuleClasses.MODULE_MAPS);
    }

    if (cppConfiguration.forcePic()) {
      if (unsupportedFeatures.contains(CppRuleClasses.SUPPORTS_PIC)) {
        throw new EvalException(PIC_CONFIGURATION_ERROR);
      }
      allRequestedFeaturesBuilder.add(CppRuleClasses.SUPPORTS_PIC);
    }

    if (cppConfiguration.appleGenerateDsym()) {
      allRequestedFeaturesBuilder.add(CppRuleClasses.GENERATE_DSYM_FILE_FEATURE_NAME);
    } else {
      allRequestedFeaturesBuilder.add(CppRuleClasses.NO_GENERATE_DEBUG_SYMBOLS_FEATURE_NAME);
    }

    if (language == Language.OBJC || language == Language.OBJCPP) {
      allRequestedFeaturesBuilder.add(CppRuleClasses.LANG_OBJC);
      if (cppConfiguration.objcGenerateLinkmap()) {
        allRequestedFeaturesBuilder.add(CppRuleClasses.GENERATE_LINKMAP_FEATURE_NAME);
      }
      if (cppConfiguration.objcShouldStripBinary()) {
        allRequestedFeaturesBuilder.add(CppRuleClasses.DEAD_STRIP_FEATURE_NAME);
      }
    }

    ImmutableSet<String> allUnsupportedFeatures = unsupportedFeaturesBuilder.build();

    ImmutableList.Builder<String> allFeatures =
        new ImmutableList.Builder<String>()
            .addAll(ImmutableSet.of(cppConfiguration.getCompilationMode().toString()))
            .addAll(DEFAULT_ACTION_CONFIGS)
            .addAll(requestedFeatures)
            .addAll(toolchain.getFeatures().getDefaultFeaturesAndActionConfigs());

    if (language == Language.OBJC || language == Language.OBJCPP) {
      allFeatures.addAll(OBJC_ACTIONS);
    }

    if (!cppConfiguration.dontEnableHostNonhost()) {
      if (toolchain.isToolConfiguration()) {
        allFeatures.add("host");
      } else {
        allFeatures.add("nonhost");
      }
    }

    allFeatures.addAll(getCoverageFeatures(cppConfiguration));

    if (!allUnsupportedFeatures.contains(CppRuleClasses.FDO_INSTRUMENT)) {
      if (cppConfiguration.getFdoInstrument() != null) {
        allFeatures.add(CppRuleClasses.FDO_INSTRUMENT);
      } else {
        if (cppConfiguration.getCSFdoInstrument() != null) {
          allFeatures.add(CppRuleClasses.CS_FDO_INSTRUMENT);
        }
      }
    }

    FdoContext.BranchFdoProfile branchFdoProvider = toolchain.getFdoContext().getBranchFdoProfile();

    boolean enablePropellerOptimize =
        (toolchain.getFdoContext().getPropellerOptimizeInputFile() != null
            && (toolchain.getFdoContext().getPropellerOptimizeInputFile().getCcArtifact() != null
                || toolchain.getFdoContext().getPropellerOptimizeInputFile().getLdArtifact()
                    != null));

    if (branchFdoProvider != null && cppConfiguration.getCompilationMode() == CompilationMode.OPT) {
      if ((branchFdoProvider.isLlvmFdo() || branchFdoProvider.isLlvmCSFdo())
          && !allUnsupportedFeatures.contains(CppRuleClasses.FDO_OPTIMIZE)) {
        allFeatures.add(CppRuleClasses.FDO_OPTIMIZE);
        // Support implicit enabling of ThinLTO for FDO unless it has been explicitly disabled.
        if (!allUnsupportedFeatures.contains(CppRuleClasses.THIN_LTO)) {
          allFeatures.add(CppRuleClasses.ENABLE_FDO_THINLTO);
        }

        // Support implicit enabling of split functions for FDO unless it has been explicitly
        // disabled
        // or propeller_optimize is used. propeller_optimize must also disable split functions as
        // they are mutually exclusive.
        if (!allUnsupportedFeatures.contains(CppRuleClasses.SPLIT_FUNCTIONS)
            && !enablePropellerOptimize) {
          allFeatures.add(CppRuleClasses.ENABLE_FDO_SPLIT_FUNCTIONS);
        }
      }
      if (branchFdoProvider.isLlvmCSFdo()) {
        allFeatures.add(CppRuleClasses.CS_FDO_OPTIMIZE);
      }
      if (branchFdoProvider.isAutoFdo()) {
        allFeatures.add(CppRuleClasses.AUTOFDO);
        // Support implicit enabling of Memprof for AFDO unless it has been disabled.
        if (!allUnsupportedFeatures.contains(CppRuleClasses.MEMPROF_OPTIMIZE)) {
          allFeatures.add(CppRuleClasses.ENABLE_AUTOFDO_MEMPROF_OPTIMIZE);
        }
        // Support implicit enabling of ThinLTO for AFDO unless it has been disabled.
        if (!allUnsupportedFeatures.contains(CppRuleClasses.THIN_LTO)) {
          allFeatures.add(CppRuleClasses.ENABLE_AFDO_THINLTO);
        }
        // Support implicit enabling of FSAFDO for AFDO unless it has been disabled.
        if (!allUnsupportedFeatures.contains(CppRuleClasses.FSAFDO)) {
          allFeatures.add(CppRuleClasses.ENABLE_FSAFDO);
          // Support implicit enabling of MFS for FSAFDO unless it has been disabled.
          // We are reusing the "ENABLE_FDO_SPLIT_FUNCTIONS" feature here.
          if (!allUnsupportedFeatures.contains(CppRuleClasses.SPLIT_FUNCTIONS)) {
            allFeatures.add(CppRuleClasses.ENABLE_FDO_SPLIT_FUNCTIONS);
          }
        }
      }
      if (branchFdoProvider.isAutoXBinaryFdo()) {
        allFeatures.add(CppRuleClasses.XBINARYFDO);
        // Support implicit enabling of ThinLTO for XFDO unless it has been explicitly disabled.
        if (!allUnsupportedFeatures.contains(CppRuleClasses.THIN_LTO)) {
          allFeatures.add(CppRuleClasses.ENABLE_XFDO_THINLTO);
        }
      }
    }
    if (cppConfiguration.getFdoPrefetchHintsLabel() != null) {
      allRequestedFeaturesBuilder.add(CppRuleClasses.FDO_PREFETCH_HINTS);
    }

    if (enablePropellerOptimize) {
      allRequestedFeaturesBuilder.add(CppRuleClasses.PROPELLER_OPTIMIZE);
    }

    for (String feature : allFeatures.build()) {
      if (!allUnsupportedFeatures.contains(feature)) {
        allRequestedFeaturesBuilder.add(feature);
      }
    }

    try {
      FeatureConfiguration featureConfiguration =
          toolchain.getFeatures().getFeatureConfiguration(allRequestedFeaturesBuilder.build());
      for (String feature : unsupportedFeatures) {
        if (featureConfiguration.isEnabled(feature)) {
          throw Starlark.errorf(
              "The C++ toolchain '%s' unconditionally implies feature '%s', which is unsupported"
                  + " by this rule. This is most likely a misconfiguration in the C++ toolchain.",
              toolchain.getCcToolchainLabel(), feature);
        }
      }
      if (cppConfiguration.forcePic()
          && !featureConfiguration.isEnabled(CppRuleClasses.PIC)
          && !featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_PIC)) {
        throw new EvalException(PIC_CONFIGURATION_ERROR);
      }
      return featureConfiguration;
    } catch (CollidingProvidesException ex) {
      throw new EvalException(ex);
    }
  }

  public static boolean isOldStarlarkApiWhiteListed(
      StarlarkRuleContext starlarkRuleContext, List<String> whitelistedPackages) {
    RuleContext context = starlarkRuleContext.getRuleContext();
    Rule rule = context.getRule();

    RuleClass ruleClass = rule.getRuleClassObject();
    Label label = ruleClass.getRuleDefinitionEnvironmentLabel();
    if (label.getRepository().getName().equals("_builtins")) {
      // always permit builtins
      return true;
    }
    if (label != null) {
      return whitelistedPackages.stream()
          .anyMatch(path -> label.getPackageFragment().toString().startsWith(path));
    }
    return false;
  }
}
