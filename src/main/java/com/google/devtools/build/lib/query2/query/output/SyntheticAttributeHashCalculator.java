// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.query2.query.output;

import com.google.common.base.Preconditions;
import com.google.common.flogger.GoogleLogger;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hasher;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeFormatter;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Contains the logic for condensing the various properties of rules that contribute to their
 * "affectedness" into a simple hash value. The resulting hash may be compared across queries to
 * tell if a rule has changed in a potentially meaningful way.
 */
class SyntheticAttributeHashCalculator {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final ThreadLocal<HashOutput> hashOutputByThread =
      ThreadLocal.withInitial(HashOutput::new);

  private SyntheticAttributeHashCalculator() {}

  @FunctionalInterface
  private interface SerializedAttributeLookup {
    @Nullable
    Build.Attribute get(Attribute attribute, int attrIndex);
  }

  /**
   * Returns a hash of various properties of a rule which might contribute to the rule's
   * "affectedness". This includes, but is not limited to, attribute values and error-state.
   *
   * @param rule The rule instance to calculate the hash for.
   * @param serializedAttributes Any available attribute which have already been serialized. This is
   *     an optimization to avoid re-serializing attributes internally.
   * @param extraDataForAttrHash Extra data to add to the hash.
   */
  static String compute(
      Rule rule,
      Map<Attribute, Build.Attribute> serializedAttributes,
      Object extraDataForAttrHash,
      HashFunction hashFunction,
      boolean includeAttributeSourceAspects,
      boolean includeStarlarkRuleEnv) {
    return compute(
        rule,
        (attribute, attrIndex) -> serializedAttributes.get(attribute),
        extraDataForAttrHash,
        hashFunction,
        includeAttributeSourceAspects,
        includeStarlarkRuleEnv);
  }

  static String computeWithIndexedAttributes(
      Rule rule,
      @Nullable Build.Attribute[] serializedAttributes,
      Object extraDataForAttrHash,
      HashFunction hashFunction,
      boolean includeAttributeSourceAspects,
      boolean includeStarlarkRuleEnv) {
    return compute(
        rule,
        (attribute, attrIndex) ->
            serializedAttributes == null ? null : serializedAttributes[attrIndex],
        extraDataForAttrHash,
        hashFunction,
        includeAttributeSourceAspects,
        includeStarlarkRuleEnv);
  }

  private static String compute(
      Rule rule,
      SerializedAttributeLookup serializedAttributes,
      Object extraDataForAttrHash,
      HashFunction hashFunction,
      boolean includeAttributeSourceAspects,
      boolean includeStarlarkRuleEnv) {
    HashOutput cachedOutput = hashOutputByThread.get();
    // A nested hash on the same thread needs its own encoder.
    HashOutput output = cachedOutput.inUse ? new HashOutput() : cachedOutput;
    boolean completed = false;
    try {
      output.begin(hashFunction);
      writeRule(
          rule,
          serializedAttributes,
          extraDataForAttrHash,
          includeAttributeSourceAspects,
          includeStarlarkRuleEnv,
          output.codedOut);
      String hash = output.finish();
      completed = true;
      return hash;
    } finally {
      if (!completed && output == cachedOutput) {
        // The encoder may still contain bytes from the failed hash.
        hashOutputByThread.remove();
      }
      output.clear();
    }
  }

  private static void writeRule(
      Rule rule,
      SerializedAttributeLookup serializedAttributes,
      Object extraDataForAttrHash,
      boolean includeAttributeSourceAspects,
      boolean includeStarlarkRuleEnv,
      CodedOutputStream codedOut) {
    RuleClass ruleClass = rule.getRuleClassObject();
    if (ruleClass.isStarlark() && includeStarlarkRuleEnv) {
      try {
        codedOut.writeByteArrayNoTag(
            Preconditions.checkNotNull(ruleClass.getRuleDefinitionEnvironmentDigest(), rule));
      } catch (IOException e) {
        throw new IllegalStateException("Unexpected IO failure writing to digest stream", e);
      }
    }

    RawAttributeMapper rawAttributeMapper = RawAttributeMapper.of(rule);
    var attributes = rule.getAttributeProvider().getAttributes();
    for (int attrIndex = 0; attrIndex < attributes.size(); attrIndex++) {
      Attribute attr = attributes.get(attrIndex);
      String attrName = attr.getName();

      if (attrName.equals("generator_location")) {
        // generator_location can be ignored for the purpose of telling if a rule has changed.
        continue;
      }

      Object valueToHash = rawAttributeMapper.getRawAttributeValue(rule, attr);

      if (valueToHash instanceof ComputedDefault computedDefault) {
        // ConfiguredDefaults need special handling to detect changes in evaluated values.
        if (!computedDefault.dependencies().isEmpty()) {
          // TODO(b/29038463): We're skipping computed defaults that depend on other configurable
          // attributes because there currently isn't a way to evaluate such a computed default;
          // there isn't *one* value it evaluates to.
          continue;
        }

        try {
          valueToHash = computedDefault.getDefault(rawAttributeMapper);
        } catch (IllegalArgumentException e) {
          // TODO(mschaller): Catching IllegalArgumentException isn't ideal. It's thrown by
          // AbstractAttributeMapper#get if the attribute's type doesn't match its value, which
          // would happen if a ComputedDefault function accessed an attribute whose value was
          // configurable. We check whether the ComputedDefault declared any configurable
          // attribute dependencies above, but someone could make a mistake and fail to declare
          // something. There's no mechanism that enforces correct declaration right now.
          // This allows us to recover from such an error by skipping an attribute, as opposed to
          // crashing.
          logger.atWarning().log(
              "Recovering from failed evaluation of ComputedDefault attribute value: %s", e);
          continue;
        }
      }

      Build.Attribute attrPb =
          valueToHash instanceof SelectorList<?> ? null : serializedAttributes.get(attr, attrIndex);
      if (attrPb == null) {
        // We didn't already serialize the attribute or it's a SelectorList. Latter may
        // have been flattened while we want the full representation, so we start from scratch.
        attrPb =
            AttributeFormatter.getAttributeProto(
                attr,
                valueToHash,
                /* explicitlySpecified= */ false, // We care about value, not how it was set.
                /* encodeBooleanAndTriStateAsIntegerAndString= */ false,
                /* sourceAspect= */ null,
                includeAttributeSourceAspects,
                LabelPrinter.legacy());
      }

      try {
        attrPb.writeTo(codedOut);
      } catch (IOException e) {
        throw new IllegalStateException("Unexpected IO failure writing to digest stream", e);
      }
    }

    try {
      // Rules can be considered changed when the containing package goes in/out of error.
      codedOut.writeBoolNoTag(rule.getPackageoid().containsErrors());
    } catch (IOException e) {
      throw new IllegalStateException("Unexpected IO failure writing to digest stream", e);
    }

    try {
      // Include a summary of any package-wide data that applies to this target (e.g. custom make
      // variables aka `vardef`).
      codedOut.writeStringNoTag((String) extraDataForAttrHash);
    } catch (IOException e) {
      throw new IllegalStateException("Unexpected IO failure writing to digest stream", e);
    }

    Label transitiveVisibility = rule.getTransitiveVisibility();
    try {
      codedOut.writeBoolNoTag(transitiveVisibility != null);
      if (transitiveVisibility != null) {
        codedOut.writeStringNoTag(transitiveVisibility.getCanonicalForm());
      }
    } catch (IOException e) {
      throw new IllegalStateException("Unexpected IO failure writing to digest stream", e);
    }
  }

  private static final class HashOutput extends OutputStream {
    private final CodedOutputStream codedOut = CodedOutputStream.newInstance(this);
    private Hasher hasher;
    private boolean inUse;

    void begin(HashFunction hashFunction) {
      inUse = true;
      hasher = hashFunction.newHasher();
    }

    String finish() {
      try {
        codedOut.flush();
      } catch (IOException e) {
        throw new IllegalStateException("Unexpected flush failure", e);
      }
      return BaseEncoding.base64().encode(hasher.hash().asBytes());
    }

    void clear() {
      hasher = null;
      inUse = false;
    }

    @Override
    public void write(int value) {
      hasher.putByte((byte) value);
    }

    @Override
    public void write(byte[] bytes, int offset, int length) {
      hasher.putBytes(bytes, offset, length);
    }
  }
}
