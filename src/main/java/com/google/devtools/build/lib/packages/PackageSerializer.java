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
package com.google.devtools.build.lib.packages;

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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.MakeEnvironment.Binding;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.syntax.GlobCriteria;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Label;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Functionality to serialize loaded packages.
 */
public class PackageSerializer {
  private static Build.SourceFile serializeInputFile(InputFile inputFile) {
    Build.SourceFile.Builder result = Build.SourceFile.newBuilder();
    result.setName(inputFile.getLabel().toString());
    if (inputFile.isVisibilitySpecified()) {
      for (Label visibilityLabel : inputFile.getVisibility().getDeclaredLabels()) {
        result.addVisibilityLabel(visibilityLabel.toString());
      }
    }
    if (inputFile.isLicenseSpecified()) {
      result.setLicense(serializeLicense(inputFile.getLicense()));
    }

    result.setParseableLocation(serializeLocation(inputFile.getLocation()));
    return result.build();
  }

  private static Build.Location serializeLocation(Location location) {
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

  private static Build.PackageGroup serializePackageGroup(PackageGroup packageGroup) {
    Build.PackageGroup.Builder result = Build.PackageGroup.newBuilder();

    result.setName(packageGroup.getLabel().toString());
    result.setParseableLocation(serializeLocation(packageGroup.getLocation()));

    for (PackageSpecification packageSpecification : packageGroup.getPackageSpecifications()) {
      result.addContainedPackage(packageSpecification.toString());
    }

    for (Label include : packageGroup.getIncludes()) {
      result.addIncludedPackageGroup(include.toString());
    }

    return result.build();
  }

  private static Build.Rule serializeRule(Rule rule) {
    Build.Rule.Builder result = Build.Rule.newBuilder();
    result.setName(rule.getLabel().toString());
    result.setRuleClass(rule.getRuleClass());
    result.setParseableLocation(serializeLocation(rule.getLocation()));
    for (Attribute attribute : rule.getAttributes()) {
      Object value = attribute.getName().equals("visibility")
          ? rule.getVisibility().getDeclaredLabels()
          // TODO(bazel-team): support configurable attributes. AggregatingAttributeMapper
          // may make more sense here. Computed defaults may add complications.
          : RawAttributeMapper.of(rule).get(attribute.getName(), attribute.getType());
      if (value != null) {
        PackageSerializer.addAttributeToProto(result, attribute, value,
            rule.getAttributeLocation(attribute.getName()),
            rule.isAttributeValueExplicitlySpecified(attribute),
            true);
      }
    }

    return result.build();
  }

  private static List<Build.MakeVar> serializeMakeEnvironment(MakeEnvironment makeEnv) {
    List<Build.MakeVar> result = new ArrayList<>();

    for (Map.Entry<String, ImmutableList<Binding>> var : makeEnv.getBindings().entrySet()) {
      Build.MakeVar.Builder varPb = Build.MakeVar.newBuilder();
      varPb.setName(var.getKey());
      for (Binding binding : var.getValue()) {
        Build.MakeVarBinding.Builder bindingPb = Build.MakeVarBinding.newBuilder();
        bindingPb.setValue(binding.getValue());
        bindingPb.setPlatformSetRegexp(binding.getPlatformSetRegexp());
        varPb.addBinding(bindingPb);
      }

      result.add(varPb.build());
    }

    return result;
  }

  private static Build.License serializeLicense(License license) {
    Build.License.Builder result = Build.License.newBuilder();

    for (License.LicenseType licenseType : license.getLicenseTypes()) {
      result.addLicenseType(licenseType.toString());
    }

    for (Label exception : license.getExceptions()) {
      result.addException(exception.toString());
    }
    return result.build();
  }

  private static Build.Event serializeEvent(Event event) {
    Build.Event.Builder result = Build.Event.newBuilder();
    result.setMessage(event.getMessage());
    if (event.getLocation() != null) {
      result.setLocation(serializeLocation(event.getLocation()));
    }

    Build.Event.EventKind kind;

    switch (event.getKind()) {
      case ERROR:
        kind = Build.Event.EventKind.ERROR;
        break;
      case WARNING:
        kind = Build.Event.EventKind.WARNING;
        break;
      case INFO:
        kind = Build.Event.EventKind.INFO;
        break;
      case PROGRESS:
        kind = Build.Event.EventKind.PROGRESS;
        break;
      default: throw new IllegalStateException();
    }

    result.setKind(kind);
    return result.build();
  }

  private static void serializePackageInternal(Package pkg, Build.Package.Builder builder) {
    builder.setName(pkg.getName());
    builder.setRepository(pkg.getPackageIdentifier().getRepository().toString());
    builder.setBuildFilePath(pkg.getFilename().getPathString());
    // The extra bit is needed to handle the corner case when the default visibility is [], i.e.
    // zero labels.
    builder.setDefaultVisibilitySet(pkg.isDefaultVisibilitySet());
    if (pkg.isDefaultVisibilitySet()) {
      for (Label visibilityLabel : pkg.getDefaultVisibility().getDeclaredLabels()) {
        builder.addDefaultVisibilityLabel(visibilityLabel.toString());
      }
    }

    builder.setDefaultTestonly(pkg.getDefaultTestOnly());
    if (pkg.getDefaultDeprecation() != null) {
      builder.setDefaultDeprecation(pkg.getDefaultDeprecation());
    }

    for (String defaultCopt : pkg.getDefaultCopts()) {
      builder.addDefaultCopt(defaultCopt);
    }

    if (pkg.isDefaultHdrsCheckSet()) {
      builder.setDefaultHdrsCheck(pkg.getDefaultHdrsCheck());
    }

    builder.setDefaultLicense(serializeLicense(pkg.getDefaultLicense()));

    for (DistributionType distributionType : pkg.getDefaultDistribs()) {
      builder.addDefaultDistrib(distributionType.toString());
    }

    for (String feature : pkg.getFeatures()) {
      builder.addDefaultSetting(feature);
    }

    for (Label subincludeLabel : pkg.getSubincludeLabels()) {
      builder.addSubincludeLabel(subincludeLabel.toString());
    }

    for (Label skylarkLabel : pkg.getSkylarkFileDependencies()) {
      builder.addSkylarkLabel(skylarkLabel.toString());
    }

    for (Build.MakeVar makeVar :
         serializeMakeEnvironment(pkg.getMakeEnvironment())) {
      builder.addMakeVariable(makeVar);
    }

    for (Target target : pkg.getTargets()) {
      if (target instanceof InputFile) {
        builder.addSourceFile(serializeInputFile((InputFile) target));
      } else if (target instanceof OutputFile) {
        // Output files are ignored; they are recorded in rules.
      } else if (target instanceof PackageGroup) {
        builder.addPackageGroup(serializePackageGroup((PackageGroup) target));
      } else if (target instanceof Rule) {
        builder.addRule(serializeRule((Rule) target));
      }
    }

    for (Event event : pkg.getEvents()) {
      builder.addEvent(serializeEvent(event));
    }

    builder.setContainsErrors(pkg.containsErrors());
    builder.setContainsTemporaryErrors(pkg.containsTemporaryErrors());
  }

  /**
   * Serialize a package to a protocol message. The inverse of
   * {@link PackageDeserializer#deserialize}.
   */
  public static Build.Package serializePackage(Package pkg) {
    Build.Package.Builder builder = Build.Package.newBuilder();
    serializePackageInternal(pkg, builder);
    return builder.build();
  }

  /**
   * Adds the serialized version of the specified attribute to the specified message.
   *
   * @param result the message to amend
   * @param attr the attribute to add
   * @param value the value of the attribute
   * @param location the location of the attribute in the source file
   * @param explicitlySpecified whether the attribute was explicitly specified or not
   * @param includeGlobs add glob expression for attributes that contain them
   */
  // TODO(bazel-team): This is a copy of the code in ProtoOutputFormatter.
  @SuppressWarnings("unchecked")
  public static void addAttributeToProto(
      Build.Rule.Builder result, Attribute attr, Object value, Location location,
      Boolean explicitlySpecified, boolean includeGlobs) {
    // Get the attribute type.  We need to convert and add appropriately
    com.google.devtools.build.lib.packages.Type<?> type = attr.getType();

    Build.Attribute.Builder attrPb = Build.Attribute.newBuilder();

    // Set the type, name and source
    Build.Attribute.Discriminator attributeProtoType = ProtoUtils.getDiscriminatorFromType(type);
    attrPb.setName(attr.getName());
    attrPb.setType(attributeProtoType);

    if (location != null) {
      attrPb.setParseableLocation(serializeLocation(location));
    }

    if (explicitlySpecified != null) {
      attrPb.setExplicitlySpecified(explicitlySpecified);
    }

    /*
     * Set the appropriate type and value.  Since string and string list store
     * values for multiple types, use the toString() method on the objects
     * instead of casting them.  Note that Boolean and TriState attributes have
     * both an integer and string representation.
     */
    if (type == INTEGER) {
      attrPb.setIntValue((Integer) value);
    } else if (type == STRING || type == LABEL || type == NODEP_LABEL || type == OUTPUT) {
      attrPb.setStringValue(value.toString());
    } else if (type == STRING_LIST || type == LABEL_LIST || type == NODEP_LABEL_LIST
        || type == OUTPUT_LIST || type == DISTRIBUTIONS) {
      Collection<?> values = (Collection<?>) value;
      for (Object entry : values) {
        attrPb.addStringListValue(entry.toString());
      }
    } else if (type == INTEGER_LIST) {
      Collection<Integer> values = (Collection<Integer>) value;
      for (Integer entry : values) {
        attrPb.addIntListValue(entry);
      }
    } else if (type == BOOLEAN) {
      if ((Boolean) value) {
        attrPb.setStringValue("true");
        attrPb.setBooleanValue(true);
      } else {
        attrPb.setStringValue("false");
        attrPb.setBooleanValue(false);
      }
      // This maintains partial backward compatibility for external users of the
      // protobuf that were expecting an integer field and not a true boolean.
      attrPb.setIntValue((Boolean) value ? 1 : 0);
    } else if (type == TRISTATE) {
      switch ((TriState) value) {
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
      }
    } else if (type == LICENSE) {
      License license = (License) value;
      Build.License.Builder licensePb = Build.License.newBuilder();
      for (License.LicenseType licenseType : license.getLicenseTypes()) {
        licensePb.addLicenseType(licenseType.toString());
      }
      for (Label exception : license.getExceptions()) {
        licensePb.addException(exception.toString());
      }
      attrPb.setLicense(licensePb);
    } else if (type == STRING_DICT) {
      Map<String, String> dict = (Map<String, String>) value;
      for (Map.Entry<String, String> dictEntry : dict.entrySet()) {
        Build.StringDictEntry entry = Build.StringDictEntry.newBuilder()
            .setKey(dictEntry.getKey())
            .setValue(dictEntry.getValue())
            .build();
        attrPb.addStringDictValue(entry);
      }
    } else if (type == STRING_DICT_UNARY) {
      Map<String, String> dict = (Map<String, String>) value;
      for (Map.Entry<String, String> dictEntry : dict.entrySet()) {
        Build.StringDictUnaryEntry entry = Build.StringDictUnaryEntry.newBuilder()
            .setKey(dictEntry.getKey())
            .setValue(dictEntry.getValue())
            .build();
        attrPb.addStringDictUnaryValue(entry);
      }
    } else if (type == STRING_LIST_DICT) {
      Map<String, List<String>> dict = (Map<String, List<String>>) value;
      for (Map.Entry<String, List<String>> dictEntry : dict.entrySet()) {
        Build.StringListDictEntry.Builder entry = Build.StringListDictEntry.newBuilder()
            .setKey(dictEntry.getKey());
        for (Object dictEntryValue : dictEntry.getValue()) {
          entry.addValue(dictEntryValue.toString());
        }
        attrPb.addStringListDictValue(entry);
      }
    } else if (type == LABEL_LIST_DICT) {
      Map<String, List<Label>> dict = (Map<String, List<Label>>) value;
      for (Map.Entry<String, List<Label>> dictEntry : dict.entrySet()) {
        Build.LabelListDictEntry.Builder entry = Build.LabelListDictEntry.newBuilder()
            .setKey(dictEntry.getKey());
        for (Object dictEntryValue : dictEntry.getValue()) {
          entry.addValue(dictEntryValue.toString());
        }
        attrPb.addLabelListDictValue(entry);
      }
    } else if (type == FILESET_ENTRY_LIST) {
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
    } else {
      throw new IllegalStateException("Unknown type: " + type);
    }

    if (includeGlobs && value instanceof GlobList<?>) {
      GlobList<?> globList = (GlobList<?>) value;

      for (GlobCriteria criteria : globList.getCriteria()) {
        Build.GlobCriteria.Builder criteriaPb = Build.GlobCriteria.newBuilder();
        criteriaPb.setGlob(criteria.isGlob());
        for (String include : criteria.getIncludePatterns()) {
          criteriaPb.addInclude(include);
        }
        for (String exclude : criteria.getExcludePatterns()) {
          criteriaPb.addExclude(exclude);
        }

        attrPb.addGlobCriteria(criteriaPb);
      }
    }
    result.addAttribute(attrPb);
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
}
