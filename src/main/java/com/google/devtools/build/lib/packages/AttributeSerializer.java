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
package com.google.devtools.build.lib.packages;

import static com.google.devtools.build.lib.packages.BuildType.DISTRIBUTIONS;
import static com.google.devtools.build.lib.packages.BuildType.FILESET_ENTRY_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_DICT_UNARY;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST_DICT;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.OUTPUT;
import static com.google.devtools.build.lib.packages.BuildType.OUTPUT_LIST;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.INTEGER;
import static com.google.devtools.build.lib.syntax.Type.INTEGER_LIST;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_DICT;
import static com.google.devtools.build.lib.syntax.Type.STRING_DICT_UNARY;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST_DICT;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.syntax.GlobCriteria;
import com.google.devtools.build.lib.syntax.GlobList;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/** Common utilities for serializing {@link Attribute}s as protocol buffers. */
public class AttributeSerializer {

  private AttributeSerializer() {}

  /**
   * Returns the possible values of the specified attribute in the specified rule. For
   * non-configured attributes, this is a single value. For configurable attributes, this
   * may be multiple values.
   */
  public static Iterable<Object> getAttributeValues(Rule rule, Attribute attr) {
    // Values may be null, so use normal collections rather than immutable collections.
    if (attr.getName().equals("visibility")) {
      List<Object> result = new ArrayList<>(1);
      result.add(rule.getVisibility().getDeclaredLabels());
      return result;
    } else {
      return Lists.<Object>newArrayList(
          AggregatingAttributeMapper.of(rule).visitAttribute(attr.getName(), attr.getType()));
    }
  }

  /**
   * Convert Attribute to proto representation. If {@code includeGlobs} is true then include
   * globs expressions when present, omit otherwise.
   */
  @SuppressWarnings("unchecked")
  public static Build.Attribute getAttributeProto(
      Attribute attr, Iterable<Object> values, boolean explicitlySpecified, boolean includeGlobs) {
    // Get the attribute type.  We need to convert and add appropriately
    com.google.devtools.build.lib.syntax.Type<?> type = attr.getType();

    Build.Attribute.Builder attrPb = Build.Attribute.newBuilder();

    // Set the type, name and source
    attrPb.setName(attr.getName());
    attrPb.setType(ProtoUtils.getDiscriminatorFromType(type));
    attrPb.setExplicitlySpecified(explicitlySpecified);

    // Convenience binding for single-value attributes. Because those attributes can only
    // have a single value, when we encounter configurable versions of them we need to
    // react somehow to having multiple possible values to report. We currently just
    // refrain from setting *any* value in that scenario. This variable is set to null
    // to indicate that.
    //
    // For example, for "linkstatic = select({':foo': 0, ':bar': 1})", "values" will contain [0, 1].
    // Since linkstatic is a single-value string element, its proto field (string_value) can't
    // store both values. Since no use case today actually needs this, we just skip it.
    //
    // TODO(bazel-team): support this properly. This will require syntactic change to build.proto
    // (or reinterpretation of its current fields).
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
      attrPb.setNodep(type == NODEP_LABEL);
    } else if (type == STRING_LIST || type == LABEL_LIST || type == NODEP_LABEL_LIST
        || type == OUTPUT_LIST || type == DISTRIBUTIONS) {
      for (Object value : values) {
        for (Object entry : (Collection<?>) value) {
          attrPb.addStringListValue(entry.toString());
        }
      }
      attrPb.setNodep(type == NODEP_LABEL_LIST);
    } else if (type == INTEGER_LIST) {
      for (Object value : values) {
        for (Integer entry : (Collection<Integer>) value) {
          attrPb.addIntListValue(entry);
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
            throw new AssertionError("Expected AUTO/NO/YES to cover all possible cases");
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
        Map<String, String> dict = (Map<String, String>) value;
        for (Map.Entry<String, String> keyValueList : dict.entrySet()) {
          Build.StringDictEntry entry = Build.StringDictEntry.newBuilder()
              .setKey(keyValueList.getKey())
              .setValue(keyValueList.getValue())
              .build();
          attrPb.addStringDictValue(entry);
        }
      }
    } else if (type == STRING_DICT_UNARY) {
      for (Object value : values) {
        Map<String, String> dict = (Map<String, String>) value;
        for (Map.Entry<String, String> dictEntry : dict.entrySet()) {
          Build.StringDictUnaryEntry entry = Build.StringDictUnaryEntry.newBuilder()
              .setKey(dictEntry.getKey())
              .setValue(dictEntry.getValue())
              .build();
          attrPb.addStringDictUnaryValue(entry);
        }
      }
    } else if (type == STRING_LIST_DICT) {
      for (Object value : values) {
        Map<String, List<String>> dict = (Map<String, List<String>>) value;
        for (Map.Entry<String, List<String>> dictEntry : dict.entrySet()) {
          Build.StringListDictEntry.Builder entry = Build.StringListDictEntry.newBuilder()
              .setKey(dictEntry.getKey());
          for (Object dictEntryValue : dictEntry.getValue()) {
            entry.addValue(dictEntryValue.toString());
          }
          attrPb.addStringListDictValue(entry);
        }
      }
    } else if (type == LABEL_DICT_UNARY) {
      for (Object value : values) {
        Map<String, Label> dict = (Map<String, Label>) value;
        for (Map.Entry<String, Label> dictEntry : dict.entrySet()) {
          Build.LabelDictUnaryEntry entry = Build.LabelDictUnaryEntry.newBuilder()
              .setKey(dictEntry.getKey())
              .setValue(dictEntry.getValue().toString())
              .build();
          attrPb.addLabelDictUnaryValue(entry);
        }
      }
    } else if (type == LABEL_LIST_DICT) {
      for (Object value : values) {
        Map<String, List<Label>> dict = (Map<String, List<Label>>) value;
        for (Map.Entry<String, List<Label>> dictEntry : dict.entrySet()) {
          Build.LabelListDictEntry.Builder entry = Build.LabelListDictEntry.newBuilder()
              .setKey(dictEntry.getKey());
          for (Object dictEntryValue : dictEntry.getValue()) {
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
      throw new AssertionError("Unknown type: " + type);
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

    return attrPb.build();
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

