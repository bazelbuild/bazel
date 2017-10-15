// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.common.options;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * This extends IsolatedOptionsData with information that can only be determined once all the {@link
 * OptionsBase} subclasses for a parser are known. In particular, this includes expansion
 * information.
 */
@Immutable
final class OptionsData extends IsolatedOptionsData {

  /**
   * Keeps track of all the information needed to calculate expansion flags, whether they come from
   * a static list or a @{link ExpansionFunction} object.
   */
  static class ExpansionData {
    private final ImmutableList<String> staticExpansion;
    @Nullable private final ExpansionFunction dynamicExpansions;

    ExpansionData(ImmutableList<String> staticExpansion) {
      Preconditions.checkArgument(staticExpansion != null);
      this.staticExpansion = staticExpansion;
      this.dynamicExpansions = null;
    }

    ExpansionData(ExpansionFunction dynamicExpansions) {
      Preconditions.checkArgument(dynamicExpansions != null);
      this.staticExpansion = EMPTY_EXPANSION;
      this.dynamicExpansions = dynamicExpansions;
    }

    ImmutableList<String> getExpansion(ExpansionContext context) throws OptionsParsingException {
      Preconditions.checkArgument(context != null);
      if (dynamicExpansions != null) {
        ImmutableList<String> result = dynamicExpansions.getExpansion(context);
        if (result == null) {
          String valueString =
              context.getUnparsedValue() != null ? context.getUnparsedValue() : "(null)";
          String name = context.getOptionDefinition().getOptionName();
          throw new OptionsParsingException(
              "Error expanding option '"
                  + name
                  + "': no expansions defined for value: "
                  + valueString,
              name);
        }
        return result;
      } else {
        return staticExpansion;
      }
    }

    boolean isEmpty() {
      return staticExpansion.isEmpty() && (dynamicExpansions == null);
    }
  }

  /**
   * Mapping from each Option-annotated field with expansion information to the {@link
   * ExpansionData} needed to caclulate it.
   */
  private final ImmutableMap<OptionDefinition, ExpansionData> expansionDataForFields;

  /** Construct {@link OptionsData} by extending an {@link IsolatedOptionsData} with new info. */
  private OptionsData(
      IsolatedOptionsData base, Map<OptionDefinition, ExpansionData> expansionDataForFields) {
    super(base);
    this.expansionDataForFields = ImmutableMap.copyOf(expansionDataForFields);
  }

  private static final ImmutableList<String> EMPTY_EXPANSION = ImmutableList.<String>of();
  private static final ExpansionData EMPTY_EXPANSION_DATA = new ExpansionData(EMPTY_EXPANSION);

  /**
   * Returns the expansion of an options field, regardless of whether it was defined using {@link
   * Option#expansion} or {@link Option#expansionFunction}. If the field is not an expansion option,
   * returns an empty array.
   */
  public ImmutableList<String> getEvaluatedExpansion(
      OptionDefinition optionDefinition, @Nullable String unparsedValue)
      throws OptionsParsingException {
    ExpansionData expansionData = expansionDataForFields.get(optionDefinition);
    if (expansionData == null) {
      return EMPTY_EXPANSION;
    }

    return expansionData.getExpansion(new ExpansionContext(this, optionDefinition, unparsedValue));
  }

  ExpansionData getExpansionDataForField(OptionDefinition optionDefinition) {
    ExpansionData result = expansionDataForFields.get(optionDefinition);
    return result != null ? result : EMPTY_EXPANSION_DATA;
  }

  /**
   * Constructs an {@link OptionsData} object for a parser that knows about the given {@link
   * OptionsBase} classes. In addition to the work done to construct the {@link
   * IsolatedOptionsData}, this also computes expansion information. If an option has static
   * expansions or uses an expansion function that takes a Void object, try to precalculate the
   * expansion here.
   */
  static OptionsData from(Collection<Class<? extends OptionsBase>> classes) {
    IsolatedOptionsData isolatedData = IsolatedOptionsData.from(classes);

    // All that's left is to compute expansions.
    ImmutableMap.Builder<OptionDefinition, ExpansionData> expansionDataBuilder =
        ImmutableMap.<OptionDefinition, ExpansionData>builder();
    for (Map.Entry<String, OptionDefinition> entry : isolatedData.getAllNamedFields()) {
      OptionDefinition optionDefinition = entry.getValue();
      // Determine either the hard-coded expansion, or the ExpansionFunction class. The
      // OptionProcessor checks at compile time that these aren't used together.
      String[] constExpansion = optionDefinition.getOptionExpansion();
      Class<? extends ExpansionFunction> expansionFunctionClass =
          optionDefinition.getExpansionFunction();
      if (constExpansion.length > 0) {
        expansionDataBuilder.put(
            optionDefinition, new ExpansionData(ImmutableList.copyOf(constExpansion)));
      } else if (optionDefinition.usesExpansionFunction()) {
        if (Modifier.isAbstract(expansionFunctionClass.getModifiers())) {
          throw new AssertionError(
              "The expansionFunction type " + expansionFunctionClass + " must be a concrete type");
        }
        // Evaluate the ExpansionFunction.
        ExpansionFunction instance;
        try {
          Constructor<?> constructor = expansionFunctionClass.getConstructor();
          instance = (ExpansionFunction) constructor.newInstance();
        } catch (Exception e) {
          // This indicates an error in the ExpansionFunction, and should be discovered the first
          // time it is used.
          throw new AssertionError(e);
        }

        ImmutableList<String> staticExpansion;
        try {
          staticExpansion =
              instance.getExpansion(new ExpansionContext(isolatedData, optionDefinition, null));
          Preconditions.checkState(
              staticExpansion != null,
              "Error calling expansion function for option: %s",
              optionDefinition.getOptionName());
          expansionDataBuilder.put(optionDefinition, new ExpansionData(staticExpansion));
        } catch (ExpansionNeedsValueException e) {
          // This expansion function needs data that isn't available yet. Save the instance and call
          // it later.
          expansionDataBuilder.put(optionDefinition, new ExpansionData(instance));
        } catch (OptionsParsingException e) {
          throw new IllegalStateException("Error expanding void expansion function: ", e);
        }
      }
    }
    return new OptionsData(isolatedData, expansionDataBuilder.build());
  }
}
