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

package com.google.devtools.build.lib.pkgcache;


import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;

/**
 * Options for configuring the PackageCache.
 */
public class PackageCacheOptions extends OptionsBase {

  /**
   * Converter for the {@code --default_visibility} option.
   */
  public static class DefaultVisibilityConverter implements Converter<RuleVisibility> {
    @Override
    public RuleVisibility convert(String input) throws OptionsParsingException {
      if (input.equals("public")) {
        return ConstantRuleVisibility.PUBLIC;
      } else if (input.equals("private")) {
        return ConstantRuleVisibility.PRIVATE;
      } else {
        throw new OptionsParsingException("Not a valid default visibility: '" + input
            + "' (should be 'public' or 'private'");
      }
    }

    @Override
    public String getTypeDescription() {
      return "default visibility";
    }
  }

  @Option(
      name = "package_path",
      defaultValue = "%workspace%",
      converter = Converters.ColonSeparatedOptionListConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "A colon-separated list of where to look for packages. "
              + "Elements beginning with '%workspace%' are relative to the enclosing "
              + "workspace. If omitted or empty, the default is the output of "
              + "'bazel info default-package-path'.")
  public List<String> packagePath;

  @Option(
      name = "show_loading_progress",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "If enabled, causes Bazel to print \"Loading package:\" messages.")
  public boolean showLoadingProgress;

  @Option(
    name = "deleted_packages",
    defaultValue = "",
    converter = CommaSeparatedPackageNameListConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "A comma-separated list of names of packages which the "
            + "build system will consider non-existent, even if they are "
            + "visible somewhere on the package path.\n"
            + "Use this option when deleting a subpackage 'x/y' of an "
            + "existing package 'x'.  For example, after deleting x/y/BUILD "
            + "in your client, the build system may complain if it "
            + "encounters a label '//x:y/z' if that is still provided by another "
            + "package_path entry.  Specifying --deleted_packages x/y avoids this "
            + "problem."
  )
  public List<PackageIdentifier> deletedPackages;

  @Option(
    name = "default_visibility",
    defaultValue = "private",
    converter = DefaultVisibilityConverter.class,
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Default visibility for packages that don't set it explicitly ('public' or " + "'private')."
  )
  public RuleVisibility defaultVisibility;

  @Option(
    name = "legacy_globbing_threads",
    defaultValue = "100",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Number of threads to use for glob evaluation."
  )
  public int globbingThreads;

  @Option(
    name = "experimental_max_directories_to_eagerly_visit_in_globbing",
    defaultValue = "-1",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If non-negative, the first time a glob is evaluated in a package, the subdirectories of "
            + "the package will be traversed in order to warm filesystem caches and compensate for "
            + "lack of parallelism in globbing. At most this many directories will be visited."
  )
  public int maxDirectoriesToEagerlyVisitInGlobbing;

  @Option(
    name = "fetch",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Allows the command to fetch external dependencies"
  )
  public boolean fetch;

  @Option(
    name = "experimental_check_output_files",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Check for modifications made to the output files of a build. Consider setting "
            + "this flag to false to see the effect on incremental build times."
  )
  public boolean checkOutputFiles;

  /**
   * A converter from strings containing comma-separated names of packages to lists of strings.
   */
  public static class CommaSeparatedPackageNameListConverter
      implements Converter<List<PackageIdentifier>> {

    private static final Splitter COMMA_SPLITTER = Splitter.on(',');

    @Override
    public List<PackageIdentifier> convert(String input) throws OptionsParsingException {
      if (Strings.isNullOrEmpty(input)) {
        return ImmutableList.of();
      }
      ImmutableList.Builder<PackageIdentifier> list = ImmutableList.builder();
      for (String s : COMMA_SPLITTER.split(input)) {
        try {
          list.add(PackageIdentifier.parse(s));
        } catch (LabelSyntaxException e) {
          throw new OptionsParsingException(e.getMessage());
        }
      }
      return list.build();
    }

    @Override
    public String getTypeDescription() {
      return "comma-separated list of package names";
    }

  }

  public ImmutableSet<PackageIdentifier> getDeletedPackages() {
    if (deletedPackages == null || deletedPackages.isEmpty()) {
      return ImmutableSet.of();
    }
    ImmutableSet.Builder<PackageIdentifier> newDeletedPackages = ImmutableSet.builder();
    for (PackageIdentifier pkg : deletedPackages) {
      newDeletedPackages.add(pkg.makeAbsolute());
    }
    return newDeletedPackages.build();
  }
}
