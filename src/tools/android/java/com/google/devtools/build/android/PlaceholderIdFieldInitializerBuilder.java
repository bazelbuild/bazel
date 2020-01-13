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
package com.google.devtools.build.android;

import com.android.resources.ResourceType;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.devtools.build.android.AndroidFrameworkAttrIdProvider.AttrLookupException;
import com.google.devtools.build.android.resources.FieldInitializer;
import com.google.devtools.build.android.resources.FieldInitializers;
import com.google.devtools.build.android.resources.IntArrayFieldInitializer;
import com.google.devtools.build.android.resources.IntFieldInitializer;
import com.google.devtools.build.android.resources.Visibility;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.logging.Logger;

/**
 * Generates {@link FieldInitializer}s placeholder unique ids. The real ids will be assigned when
 * building the android_binary.
 */
class PlaceholderIdFieldInitializerBuilder {

  private static final ImmutableList<ResourceType> INITIAL_TYPES =
      ImmutableList.of(
          ResourceType.DRAWABLE,
          ResourceType.MIPMAP,
          ResourceType.LAYOUT,
          ResourceType.ANIM,
          ResourceType.ANIMATOR,
          ResourceType.TRANSITION,
          ResourceType.INTERPOLATOR,
          ResourceType.XML,
          ResourceType.RAW);

  private static final ImmutableSet<ResourceType> SPECIALLY_HANDLED_TYPES =
      ImmutableSet.<ResourceType>builder()
          // These types should always be handled first
          .addAll(INITIAL_TYPES)
          // The ATTR and STYLEABLE types are handled by completely separate code and should not be
          // included in the ordered list of types
          .add(ResourceType.ATTR)
          .add(ResourceType.STYLEABLE)
          // The MENU type should always go last
          .add(ResourceType.MENU)
          .build();

  /**
   * Determine the TT portion of the resource ID (PPTTEEEE) that aapt would have assigned. This not
   * at all alphabetical. It depends on the order in which the types are processed, and whether or
   * not previous types are present (compact). See the code in aapt Resource.cpp:buildResources().
   * There are several seemingly arbitrary and different processing orders in the function, but the
   * ordering is determined specifically by the portion at: <a
   * href="https://android.googlesource.com/platform/frameworks/base.git/+/marshmallow-release/tools/aapt/Resource.cpp#1254">
   * Resource.cpp:buildResources() </a>
   *
   * <p>where it does:
   *
   * <pre>
   *   if (drawables != NULL) { ... }
   *   if (mipmaps != NULL) { ... }
   *   if (layouts != NULL) { ... }
   * </pre>
   *
   * Numbering starts at 1 instead of 0, and ResourceType.ATTR comes before the rest.
   * ResourceType.STYLEABLE doesn't actually need a resource ID, so that is skipped. We encode the
   * ordering in the following list.
   */
  private static final ImmutableList<ResourceType> AAPT_TYPE_ORDERING =
      ImmutableList.<ResourceType>builder()
          .addAll(INITIAL_TYPES)
          // The VALUES portion
          // Technically, aapt just assigns according to declaration order in the source value.xml
          // files so it isn't really deterministic. However, the Gradle merger sorts the values.xml
          // file before invoking aapt, so use the alphabetically sorted values defined in
          // ResourceType here as well.
          .addAll(
              Arrays.stream(ResourceType.values())
                  .filter((x) -> !SPECIALLY_HANDLED_TYPES.contains(x))
                  .collect(ImmutableList.toImmutableList()))
          // Technically, file-based COLOR resources come next. If we care about complete
          // equivalence we should separate the file-based resources from value-based resources so
          // that we can number them the same way.
          .add(ResourceType.MENU)
          .build();

  private static final int APP_PACKAGE_MASK = 0x7f000000;
  private static final int ATTR_TYPE_ID = 1;
  private static final String NORMALIZED_ANDROID_PREFIX = "android_";

  private static final Logger logger =
      Logger.getLogger(PlaceholderIdFieldInitializerBuilder.class.getName());

  private static int getInitialIdForTypeId(int typeId) {
    return APP_PACKAGE_MASK | (typeId << 16);
  }

  private static int nextFreeId(int nextSlot, Set<Integer> reservedSlots) {
    // Linear search for the next free slot. This assumes that reserved <public> ids are rare.
    // Otherwise we should use a NavigableSet or some other smarter data-structure.
    while (reservedSlots.contains(nextSlot)) {
      ++nextSlot;
    }
    return nextSlot;
  }

  public static PlaceholderIdFieldInitializerBuilder from(
      AndroidFrameworkAttrIdProvider androidIdProvider) {
    return new PlaceholderIdFieldInitializerBuilder(androidIdProvider);
  }

  public static PlaceholderIdFieldInitializerBuilder from(Path androidJar) {
    return from(new AndroidFrameworkAttrIdJar(androidJar));
  }

  private final AndroidFrameworkAttrIdProvider androidIdProvider;

  private final Map<ResourceType, SortedMap<String, DependencyInfo>> innerClasses =
      new EnumMap<>(ResourceType.class);

  private final Map<String, Map<String, /*inlineable=*/ Boolean>> styleableAttrs =
      new LinkedHashMap<>();

  private PlaceholderIdFieldInitializerBuilder(AndroidFrameworkAttrIdProvider androidIdProvider) {
    this.androidIdProvider = androidIdProvider;
  }

  public void addSimpleResource(
      DependencyInfo dependencyInfo, Visibility visibility, ResourceType type, String name) {
    Object unused = visibility; // TODO(b/26297204): use 'visibility'
    innerClasses
        .computeIfAbsent(type, t -> new TreeMap<>())
        .put(normalizeName(name), dependencyInfo);
  }

  public void addStyleableResource(
      DependencyInfo dependencyInfo,
      Visibility visibility,
      FullyQualifiedName key,
      Map<FullyQualifiedName, Boolean> attrs) {
    Object unused = visibility; // TODO(b/26297204): use 'visibility'
    ResourceType type = ResourceType.STYLEABLE;
    // The configuration can play a role in sorting, but that isn't modeled yet.
    String normalizedStyleableName = normalizeName(key.name());
    addSimpleResource(dependencyInfo, visibility, type, normalizedStyleableName);
    // We should have merged styleables, so there should only be one definition per configuration.
    // However, we don't combine across configurations, so there can be a pre-existing definition.
    Map<String, Boolean> normalizedAttrs = styleableAttrs.get(normalizedStyleableName);
    if (normalizedAttrs == null) {
      // We need to maintain the original order of the attrs.
      normalizedAttrs = new LinkedHashMap<>();
      styleableAttrs.put(normalizedStyleableName, normalizedAttrs);
    }
    for (Map.Entry<FullyQualifiedName, Boolean> attrEntry : attrs.entrySet()) {
      String normalizedAttrName = normalizeAttrName(attrEntry.getKey());
      normalizedAttrs.put(normalizedAttrName, attrEntry.getValue());
    }
  }

  private Map<String, Integer> assignAttrIds(int attrTypeId) {
    // Attrs are special, since they can be defined within a declare-styleable. Those are sorted
    // after top-level definitions.
    if (!innerClasses.containsKey(ResourceType.ATTR)) {
      return ImmutableMap.of();
    }
    Map<String, Integer> attrToId =
        Maps.newLinkedHashMapWithExpectedSize(innerClasses.get(ResourceType.ATTR).size());
    // After assigning public IDs, we count up monotonically, so we don't need to track additional
    // assignedIds to avoid collisions (use an ImmutableSet to ensure we don't add more).
    Set<Integer> assignedIds = ImmutableSet.of();
    Set<String> inlineAttrs = new LinkedHashSet<>();
    Set<String> styleablesWithInlineAttrs = new TreeSet<>();
    for (Map.Entry<String, Map<String, Boolean>> styleableAttrEntry : styleableAttrs.entrySet()) {
      Map<String, Boolean> attrs = styleableAttrEntry.getValue();
      for (Map.Entry<String, Boolean> attrEntry : attrs.entrySet()) {
        if (attrEntry.getValue()) {
          inlineAttrs.add(attrEntry.getKey());
          styleablesWithInlineAttrs.add(styleableAttrEntry.getKey());
        }
      }
    }
    int nextId = nextFreeId(getInitialIdForTypeId(attrTypeId), assignedIds);
    // Technically, aapt assigns based on declaration order, but the merge should have sorted
    // the non-inline attributes, so assigning by sorted order is the same.
    SortedMap<String, ?> sortedAttrs = innerClasses.get(ResourceType.ATTR);
    for (String attr : sortedAttrs.keySet()) {
      if (!inlineAttrs.contains(attr) && !attrToId.containsKey(attr)) {
        attrToId.put(attr, nextId);
        nextId = nextFreeId(nextId + 1, assignedIds);
      }
    }
    for (String styleable : styleablesWithInlineAttrs) {
      Map<String, Boolean> attrs = styleableAttrs.get(styleable);
      for (Map.Entry<String, Boolean> attrEntry : attrs.entrySet()) {
        if (attrEntry.getValue() && !attrToId.containsKey(attrEntry.getKey())) {
          attrToId.put(attrEntry.getKey(), nextId);
          nextId = nextFreeId(nextId + 1, assignedIds);
        }
      }
    }
    return ImmutableMap.copyOf(attrToId);
  }

  public FieldInitializers build() throws AttrLookupException {
    Map<ResourceType, Collection<FieldInitializer>> initializers =
        new EnumMap<>(ResourceType.class);
    Map<ResourceType, Integer> typeIdMap = chooseTypeIds();
    Map<String, Integer> attrAssignments = assignAttrIds(typeIdMap.get(ResourceType.ATTR));
    for (Map.Entry<ResourceType, SortedMap<String, DependencyInfo>> fieldEntries :
        innerClasses.entrySet()) {
      ResourceType type = fieldEntries.getKey();
      SortedMap<String, DependencyInfo> sortedFields = fieldEntries.getValue();
      ImmutableList<FieldInitializer> fields;
      if (type == ResourceType.STYLEABLE) {
        fields = getStyleableInitializers(attrAssignments, sortedFields);
      } else if (type == ResourceType.ATTR) {
        fields = getAttrInitializers(attrAssignments, sortedFields);
      } else {
        int typeId = typeIdMap.get(type);
        fields = getResourceInitializers(typeId, sortedFields);
      }
      // The maximum number of Java fields is 2^16.
      // See the JVM reference "4.11. Limitations of the Java Virtual Machine."
      Preconditions.checkArgument(fields.size() < (1 << 16));
      initializers.put(type, fields);
    }
    return FieldInitializers.copyOf(initializers);
  }

  private Map<ResourceType, Integer> chooseTypeIds() {
    Map<ResourceType, Integer> allocatedTypeIds = new EnumMap<>(ResourceType.class);
    Set<Integer> reservedTypeSlots = ImmutableSet.copyOf(allocatedTypeIds.values());
    // ATTR always takes up slot #1, even if it isn't present.
    allocatedTypeIds.put(ResourceType.ATTR, ATTR_TYPE_ID);
    // The rest are packed after that.
    int nextTypeId = nextFreeId(ATTR_TYPE_ID + 1, reservedTypeSlots);
    for (ResourceType t : AAPT_TYPE_ORDERING) {
      if (innerClasses.containsKey(t) && !allocatedTypeIds.containsKey(t)) {
        allocatedTypeIds.put(t, nextTypeId);
        nextTypeId = nextFreeId(nextTypeId + 1, reservedTypeSlots);
      }
    }
    // Sanity check that everything has been assigned, except STYLEABLE. There shouldn't be
    // anything of type PUBLIC either (since that isn't a real resource).
    // We will need to update the list if there is a new resource type.
    for (ResourceType t : innerClasses.keySet()) {
      Preconditions.checkState(
          t == ResourceType.STYLEABLE || allocatedTypeIds.containsKey(t),
          "Resource type %s is not allocated a type ID",
          t);
    }
    return allocatedTypeIds;
  }

  private static ImmutableList<FieldInitializer> getAttrInitializers(
      Map<String, Integer> attrAssignments, SortedMap<String, DependencyInfo> sortedFields) {
    ImmutableList.Builder<FieldInitializer> initList = ImmutableList.builder();
    for (Map.Entry<String, DependencyInfo> entry : sortedFields.entrySet()) {
      String field = entry.getKey();
      DependencyInfo dependencyInfo = entry.getValue();
      int attrId = attrAssignments.get(field);
      initList.add(IntFieldInitializer.of(dependencyInfo, field, attrId));
    }
    return initList.build();
  }

  private ImmutableList<FieldInitializer> getResourceInitializers(
      int typeId, SortedMap<String, DependencyInfo> sortedFields) {
    ImmutableList.Builder<FieldInitializer> initList = ImmutableList.builder();
    Map<String, Integer> publicNameToId = new LinkedHashMap<>();
    Set<Integer> assignedIds = ImmutableSet.of();
    int resourceIds = nextFreeId(getInitialIdForTypeId(typeId), assignedIds);
    for (Map.Entry<String, DependencyInfo> entry : sortedFields.entrySet()) {
      String field = entry.getKey();
      DependencyInfo dependencyInfo = entry.getValue();
      Integer fieldValue = publicNameToId.get(field);
      if (fieldValue == null) {
        fieldValue = resourceIds;
        resourceIds = nextFreeId(resourceIds + 1, assignedIds);
      }
      initList.add(IntFieldInitializer.of(dependencyInfo, field, fieldValue));
    }
    return initList.build();
  }

  private ImmutableList<FieldInitializer> getStyleableInitializers(
      Map<String, Integer> attrAssignments, SortedMap<String, DependencyInfo> sortedFields)
      throws AttrLookupException {
    ImmutableList.Builder<FieldInitializer> initList = ImmutableList.builder();
    for (Map.Entry<String, DependencyInfo> entry : sortedFields.entrySet()) {
      String field = entry.getKey();
      DependencyInfo dependencyInfo = entry.getValue();
      Set<String> attrs = styleableAttrs.get(field).keySet();
      ImmutableMap.Builder<String, Integer> arrayInitValues = ImmutableMap.builder();
      for (String attr : attrs) {
        Integer attrId = attrAssignments.get(attr);
        if (attrId == null) {
          // It should be a framework resource, otherwise we don't know about the resource.
          if (attr.startsWith(NORMALIZED_ANDROID_PREFIX)) {
            String attrWithoutPrefix = attr.substring(NORMALIZED_ANDROID_PREFIX.length());
            attrId = androidIdProvider.getAttrId(attrWithoutPrefix);
          } else if (dependencyInfo.dependencyType() == DependencyInfo.DependencyType.DIRECT) {
            // The <declare-stylable/> is in a direct dependency; assume that we don't know about
            // the attribute because it's in a transitive dependency.  The actual ID doesn't
            // matter---this is the PlaceholderIdFieldInitializerBuilder, after all.
            attrId = 0x7FFFFFFF;
          } else {
            logger.info(
                String.format(
                    "Attribute \"%s\" of styleable \"%s\" not defined among dependencies."
                        + " Ignoring.",
                    field, attr));
            continue;
          }
        }
        arrayInitValues.put(attr, attrId);
      }
      // The styleable array should be sorted by ID value.
      // Make sure that if we have android: framework attributes, their IDs are listed first.
      ImmutableMap<String, Integer> arrayInitMap =
          arrayInitValues.orderEntriesByValue(Ordering.<Integer>natural()).build();
      initList.add(IntArrayFieldInitializer.of(dependencyInfo, field, arrayInitMap.values()));
      int index = 0;
      for (String attr : arrayInitMap.keySet()) {
        initList.add(IntFieldInitializer.of(dependencyInfo, field + "_" + attr, index));
        ++index;
      }
    }
    return initList.build();
  }

  static String normalizeAttrName(FullyQualifiedName fqn) {
    String attrName = fqn.qualifiedName();
    // In addition to ".", attributes can have ":", e.g., for "android:textColor".
    Preconditions.checkArgument(!attrName.contains("::"), "invalid name %s", attrName);
    return normalizeName(attrName).replace(':', '_');
  }

  static String normalizeName(String resourceName) {
    return resourceName.replace('.', '_');
  }
}
