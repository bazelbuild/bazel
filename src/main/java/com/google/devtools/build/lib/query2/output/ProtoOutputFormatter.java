// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.query2.output;

import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.DISTRIBUTIONS;
import static com.google.devtools.build.lib.packages.Type.FILESET_ENTRY_LIST;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.INTEGER_LIST;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST_DICT;
import static com.google.devtools.build.lib.packages.Type.LICENSE;
import static com.google.devtools.build.lib.packages.Type.NODEP_LABEL;
import static com.google.devtools.build.lib.packages.Type.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.OUTPUT;
import static com.google.devtools.build.lib.packages.Type.OUTPUT_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_DICT;
import static com.google.devtools.build.lib.packages.Type.STRING_DICT_UNARY;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST_DICT;
import static com.google.devtools.build.lib.packages.Type.TRISTATE;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.GENERATED_FILE;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.PACKAGE_GROUP;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.RULE;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.SOURCE_FILE;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.ProtoUtils;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.query2.FakeSubincludeTarget;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.syntax.GlobCriteria;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.BinaryPredicate;
import com.google.devtools.build.lib.util.Pair;

import java.io.IOException;
import java.io.PrintStream;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * An output formatter that outputs a protocol buffer representation
 * of a query result and outputs the proto bytes to the output print stream.
 * By taking the bytes and calling {@code mergeFrom()} on a
 * {@code Build.QueryResult} object the full result can be reconstructed.
 */
public class ProtoOutputFormatter extends OutputFormatter {
  private BinaryPredicate<Rule, Attribute> dependencyFilter;

  protected void setDependencyFilter(QueryOptions options) {
    this.dependencyFilter = OutputFormatter.getDependencyFilter(options);
  }

  @Override
  public String getName() {
    return "proto";
  }

  @Override
  public void output(QueryOptions options, Digraph<Target> result, PrintStream out) {
    setDependencyFilter(options);

    Build.QueryResult.Builder queryResult = Build.QueryResult.newBuilder();
    try {
      for (Target target : result.getLabels()) {
        addTarget(queryResult, target);
      }

      queryResult.build().writeTo(out);
    } catch (IOException e) {
      throw new IllegalStateException("ProtocolBuffer output failed",  e);
    }
  }

  /**
   * Add the target to the query result.
   * @param queryResult The query result that contains all rule, input and
   *   output targets.
   * @param target The query target being converted to a protocol buffer.
   */
  private void addTarget(Build.QueryResult.Builder queryResult, Target target) {
    queryResult.addTarget(toTargetProtoBuffer(target));
  }

  /**
   * Converts a logical Target object into a Target protobuffer.
   */
  protected Build.Target toTargetProtoBuffer(Target target) {
    Build.Target.Builder targetPb = Build.Target.newBuilder();

    String location = target.getLocation().print();
    if (target instanceof Rule) {
      Rule rule = (Rule) target;
      Build.Rule.Builder rulePb = Build.Rule.newBuilder()
          .setName(rule.getLabel().toString())
          .setRuleClass(rule.getRuleClass())
          .setLocation(location);

      for (Attribute attr : rule.getAttributes()) {
        addAttributeToProto(rulePb, attr, getAttributeValues(rule, attr).first, null,
            rule.isAttributeValueExplicitlySpecified(attr), false);
      }

      // Include explicit elements for all direct inputs and outputs of a rule;
      // this goes beyond what is available from the attributes above, since it
      // may also (depending on options) include implicit outputs,
      // host-configuration outputs, and default values.
      for (Label label : rule.getLabels(dependencyFilter)) {
        rulePb.addRuleInput(label.toString());
      }
      for (OutputFile outputFile : rule.getOutputFiles()) {
        Label fileLabel = outputFile.getLabel();
        rulePb.addRuleOutput(fileLabel.toString());
      }
      for (String feature : rule.getFeatures()) {
        rulePb.addDefaultSetting(feature);
      }

      targetPb.setType(RULE);
      targetPb.setRule(rulePb);
    } else if (target instanceof OutputFile) {
      OutputFile outputFile = (OutputFile) target;
      Label label = outputFile.getLabel();

      Rule generatingRule = outputFile.getGeneratingRule();
      Build.GeneratedFile output = Build.GeneratedFile.newBuilder()
          .setLocation(location)
          .setGeneratingRule(generatingRule.getLabel().toString())
          .setName(label.toString())
          .build();

      targetPb.setType(GENERATED_FILE);
      targetPb.setGeneratedFile(output);
    } else if (target instanceof InputFile) {
      InputFile inputFile = (InputFile) target;
      Label label = inputFile.getLabel();

      Build.SourceFile.Builder input = Build.SourceFile.newBuilder()
          .setLocation(location)
          .setName(label.toString());

      if (inputFile.getName().equals("BUILD")) {
        for (Label subinclude : inputFile.getPackage().getSubincludes().keySet()) {
          input.addSubinclude(subinclude.toString());
        }

        for (String feature : inputFile.getPackage().getFeatures()) {
          input.addFeature(feature);
        }
      }

      for (Label visibilityDependency : target.getVisibility().getDependencyLabels()) {
        input.addPackageGroup(visibilityDependency.toString());
      }

      for (Label visibilityDeclaration : target.getVisibility().getDeclaredLabels()) {
        input.addVisibilityLabel(visibilityDeclaration.toString());
      }

      targetPb.setType(SOURCE_FILE);
      targetPb.setSourceFile(input);
    } else if (target instanceof FakeSubincludeTarget) {
      Label label = target.getLabel();
      Build.SourceFile input = Build.SourceFile.newBuilder()
          .setLocation(location)
          .setName(label.toString())
          .build();

      targetPb.setType(SOURCE_FILE);
      targetPb.setSourceFile(input);
    } else if (target instanceof PackageGroup) {
      PackageGroup packageGroup = (PackageGroup) target;
      Build.PackageGroup.Builder packageGroupPb = Build.PackageGroup.newBuilder()
          .setName(packageGroup.getLabel().toString());
      for (String containedPackage : packageGroup.getContainedPackages()) {
        packageGroupPb.addContainedPackage(containedPackage);
      }
      for (Label include : packageGroup.getIncludes()) {
        packageGroupPb.addIncludedPackageGroup(include.toString());
      }

      targetPb.setType(PACKAGE_GROUP);
      targetPb.setPackageGroup(packageGroupPb);
    } else {
      throw new IllegalArgumentException(target.toString());
    }

    return targetPb.build();
  }

  /**
   * Adds the serialized version of the specified attribute to the specified message.
   *
   * @param rulePb the message to amend
   * @param attr the attribute to add
   * @param value the possible values of the attribute (can be a multi-value list for
   *              configurable attributes)
   * @param location the location of the attribute in the source file
   * @param explicitlySpecified whether the attribute was explicitly specified or not
   * @param includeGlobs add glob expression for attributes that contain them
   */
  @SuppressWarnings("unchecked")
  public static void addAttributeToProto(
      Build.Rule.Builder rulePb, Attribute attr, Iterable<Object> values,
      Location location, Boolean explicitlySpecified, boolean includeGlobs) {
    // Get the attribute type.  We need to convert and add appropriately
    com.google.devtools.build.lib.packages.Type<?> type = attr.getType();

    Build.Attribute.Builder attrPb = Build.Attribute.newBuilder();

    // Set the type, name and source
    attrPb.setName(attr.getName());
    attrPb.setType(ProtoUtils.getDiscriminatorFromType(type));

    if (location != null) {
      attrPb.setParseableLocation(serialize(location));
    }

    if (explicitlySpecified != null) {
      attrPb.setExplicitlySpecified(explicitlySpecified);
    }

    // Convenience binding for single-value attributes. Because those attributes can only
    // have a single value, when we encounter configurable versions of them we need to
    // react somehow to having multiple possible values to report. We currently just
    // refrain from setting *any* value in that scenario. This variable is set to null
    // to indicate that scenario.
    Object singleAttributeValue = Iterables.size(values) == 1
        ? Iterables.getOnlyElement(values)
        : null;

    /*
     * Set the appropriate type and value.  Since string and string list store
     * values for multiple types, use the toString() method on the objects
     * instead of casting them.  Note that Boolean and TriState attributes have
     * both an integer and string representation.
     */
    if (type == INTEGER) {
      if (singleAttributeValue != null) {
        attrPb.setIntValue((Integer) singleAttributeValue);
      }
    } else if (type == STRING || type == LABEL || type == NODEP_LABEL || type == OUTPUT) {
      if (singleAttributeValue != null) {
        attrPb.setStringValue(singleAttributeValue.toString());
      }
    } else if (type == STRING_LIST || type == LABEL_LIST || type == NODEP_LABEL_LIST
        || type == OUTPUT_LIST || type == DISTRIBUTIONS) {
      Set<Object> visitedValues = new HashSet<>();
      for (Object value : values) {
        for (Object entry : (Collection<?>) value) {
          if (visitedValues.add(entry)) {
            attrPb.addStringListValue(entry.toString());
          }
        }
      }
    } else if (type == INTEGER_LIST) {
      Set<Integer> visitedValues = new HashSet<>();
      for (Object value : values) {
        for (Integer entry : (Collection<Integer>) value) {
          if (visitedValues.add(entry)) {
            attrPb.addIntListValue(entry);
          }
        }
      }
    } else if (type == BOOLEAN) {
      if (singleAttributeValue != null) {
        if ((Boolean) singleAttributeValue) {
          attrPb.setStringValue("true");
          attrPb.setBooleanValue(true);
        } else {
          attrPb.setStringValue("false");
          attrPb.setBooleanValue(false);
        }
        // This maintains partial backward compatibility for external users of the
        // protobuf that were expecting an integer field and not a true boolean.
        attrPb.setIntValue((Boolean) singleAttributeValue ? 1 : 0);
      }
    } else if (type == TRISTATE) {
      if (singleAttributeValue != null) {
        switch ((TriState) singleAttributeValue) {
          case AUTO:
            attrPb.setIntValue(-1);
            attrPb.setStringValue("auto");
            attrPb.setTristateValue(Build.Attribute.Tristate.AUTO);
            break;
          case NO:
            attrPb.setIntValue(0);
            attrPb.setStringValue("no");
            attrPb.setTristateValue(Build.Attribute.Tristate.NO);
            break;
          case YES:
            attrPb.setIntValue(1);
            attrPb.setStringValue("yes");
            attrPb.setTristateValue(Build.Attribute.Tristate.YES);
            break;
          default:
            throw new IllegalStateException("Execpted AUTO/NO/YES to cover all possible cases");
        }
      }
    } else if (type == LICENSE) {
      if (singleAttributeValue != null) {
        License license = (License) singleAttributeValue;
        Build.License.Builder licensePb = Build.License.newBuilder();
        for (License.LicenseType licenseType : license.getLicenseTypes()) {
          licensePb.addLicenseType(licenseType.toString());
        }
        for (Label exception : license.getExceptions()) {
          licensePb.addException(exception.toString());
        }
        attrPb.setLicense(licensePb);
      }
    } else if (type == STRING_DICT) {
      // TODO(bazel-team): support better de-duping here and in other dictionaries.
      for (Object value : values) {
      List<List<String>> dict = (List<List<String>>) value;
        for (List<String> keyValueList : dict) {
          Build.StringDictEntry entry = Build.StringDictEntry.newBuilder()
              .setKey(keyValueList.get(0))
              .setValue(keyValueList.get(1))
              .build();
          attrPb.addStringDictValue(entry);
        }
      }
    } else if (type == STRING_DICT_UNARY) {
      for (Object value : values) {
        List<Pair<String, String>> dict = (List<Pair<String, String>>) value;
        for (Pair<String, String> dictEntry : dict) {
          Build.StringDictUnaryEntry entry = Build.StringDictUnaryEntry.newBuilder()
              .setKey(dictEntry.first)
              .setValue(dictEntry.second)
              .build();
          attrPb.addStringDictUnaryValue(entry);
        }
      }
    } else if (type == STRING_LIST_DICT) {
      for (Object value : values) {
        List<Pair<String, List<?>>> dict = (List<Pair<String, List<?>>>) value;
        for (Pair<String, List<?>> dictEntry : dict) {
          Build.StringListDictEntry.Builder entry = Build.StringListDictEntry.newBuilder()
              .setKey(dictEntry.first);
          for (Object dictEntryValue : dictEntry.second) {
            entry.addValue(dictEntryValue.toString());
          }
          attrPb.addStringListDictValue(entry);
        }
      }
    } else if (type == LABEL_LIST_DICT) {
      for (Object value : values) {
        List<Pair<String, List<?>>> dict = (List<Pair<String, List<?>>>) value;
        for (Pair<String, List<?>> dictEntry : dict) {
          Build.LabelListDictEntry.Builder entry = Build.LabelListDictEntry.newBuilder()
              .setKey(dictEntry.first);
          for (Object dictEntryValue : dictEntry.second) {
            entry.addValue(dictEntryValue.toString());
          }
          attrPb.addLabelListDictValue(entry);
        }
      }
    } else if (type == FILESET_ENTRY_LIST) {
      for (Object value : values) {
        List<FilesetEntry> filesetEntries = (List<FilesetEntry>) value;
        for (FilesetEntry filesetEntry : filesetEntries) {
          Build.FilesetEntry.Builder filesetEntryPb = Build.FilesetEntry.newBuilder()
              .setSource(filesetEntry.getSrcLabel().toString())
              .setDestinationDirectory(filesetEntry.getDestDir().getPathString())
              .setSymlinkBehavior(symlinkBehaviorToPb(filesetEntry.getSymlinkBehavior()))
              .setStripPrefix(filesetEntry.getStripPrefix())
              .setFilesPresent(filesetEntry.getFiles() != null);

          if (filesetEntry.getFiles() != null) {
            for (Label file : filesetEntry.getFiles()) {
              filesetEntryPb.addFile(file.toString());
            }
          }

          if (filesetEntry.getExcludes() != null) {
            for (String exclude : filesetEntry.getExcludes()) {
              filesetEntryPb.addExclude(exclude);
            }
          }

          attrPb.addFilesetListValue(filesetEntryPb);
        }
      }
    } else {
      throw new IllegalStateException("Unknown type: " + type);
    }

    if (includeGlobs) {
      for (Object value : values) {
        if (value instanceof GlobList<?>) {
          GlobList<?> globList = (GlobList<?>) value;

          for (GlobCriteria criteria : globList.getCriteria()) {
            Build.GlobCriteria.Builder criteriaPb = Build.GlobCriteria.newBuilder()
                .setGlob(criteria.isGlob());
            for (String include : criteria.getIncludePatterns()) {
              criteriaPb.addInclude(include);
            }
            for (String exclude : criteria.getExcludePatterns()) {
              criteriaPb.addExclude(exclude);
            }

            attrPb.addGlobCriteria(criteriaPb);
          }
        }
      }
    }

    rulePb.addAttribute(attrPb);
  }

  // This is needed because I do not want to use the SymlinkBehavior from the
  // protocol buffer all over the place, so there are two classes that do
  // essentially the same thing.
  private static Build.FilesetEntry.SymlinkBehavior symlinkBehaviorToPb(
      FilesetEntry.SymlinkBehavior symlinkBehavior) {
    switch (symlinkBehavior) {
      case COPY:
        return Build.FilesetEntry.SymlinkBehavior.COPY;
      case DEREFERENCE:
        return Build.FilesetEntry.SymlinkBehavior.DEREFERENCE;
      default:
        throw new AssertionError("Unhandled FilesetEntry.SymlinkBehavior");
    }
  }

  private static Build.Location serialize(Location location) {
    Build.Location.Builder result = Build.Location.newBuilder();

    result.setStartOffset(location.getStartOffset());
    if (location.getStartLineAndColumn() != null) {
      result.setStartLine(location.getStartLineAndColumn().getLine());
      result.setStartColumn(location.getStartLineAndColumn().getColumn());
    }

    result.setEndOffset(location.getEndOffset());
    if (location.getEndLineAndColumn() != null) {
      result.setEndLine(location.getEndLineAndColumn().getLine());
      result.setEndColumn(location.getEndLineAndColumn().getColumn());
    }

    return result.build();
  }
}
