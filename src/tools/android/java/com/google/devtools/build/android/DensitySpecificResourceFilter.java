// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;

import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;

/**
 * Filters a {@link MergedAndroidData} resource drawables to the specified densities.
 */
public class DensitySpecificResourceFilter {
  private static class ResourceInfo {
    /** Path to an actual file resource, instead of a directory. */
    private Path resource;
    private String restype;
    private String qualifiers;
    private String density;
    private String resid;

    public ResourceInfo(Path resource, String restype, String qualifiers, String density,
        String resid) {
      this.resource = resource;
      this.restype = restype;
      this.qualifiers = qualifiers;
      this.density = density;
      this.resid = resid;
    }

    public Path getResource() {
      return this.resource;
    }

    public String getRestype() {
      return this.restype;
    }

    public String getQualifiers() {
      return this.qualifiers;
    }

    public String getDensity() {
      return this.density;
    }

    public String getResid() {
      return this.resid;
    }
  }

  private static class RecursiveFileCopier extends SimpleFileVisitor<Path> {
    private final Path copyToPath;
    private final List<Path> copiedSourceFiles = new ArrayList<>();
    private Path root;

    public RecursiveFileCopier(final Path copyToPath, final Path root) {
      this.copyToPath = copyToPath;
      this.root = root;
    }

    @Override
    public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
      Path copyTo = copyToPath.resolve(root.relativize(path));
      Files.createDirectories(copyTo.getParent());
      Files.copy(path, copyTo, LinkOption.NOFOLLOW_LINKS);
      copiedSourceFiles.add(copyTo);
      return FileVisitResult.CONTINUE;
    }

    public List<Path> getCopiedFiles() {
      return copiedSourceFiles;
    }
  }

  private final List<String> densities;
  private final Path out;
  private final Path working;

  private static final Map<String, Integer> DENSITY_MAP =
      new ImmutableMap.Builder<String, Integer>()
          .put("nodpi", 0)
          .put("ldpi", 120)
          .put("mdpi", 160)
          .put("tvdpi", 213)
          .put("hdpi", 240)
          .put("280dpi", 280)
          .put("xhdpi", 320)
          .put("400dpi", 400)
          .put("420dpi", 420)
          .put("xxhdpi", 480)
          .put("560dpi", 560)
          .put("xxxhdpi", 640)
          .build();

  private static final Function<ResourceInfo, String> GET_RESOURCE_ID =
      new Function<ResourceInfo, String>() {
        @Override
        public String apply(ResourceInfo info) {
          return info.getResid();
        }
      };

  private static final Function<ResourceInfo, String> GET_RESOURCE_QUALIFIERS =
      new Function<ResourceInfo, String>() {
        @Override
        public String apply(ResourceInfo info) {
          return info.getQualifiers();
        }
      };

  private static final Function<ResourceInfo, Path> GET_RESOURCE_PATH =
      new Function<ResourceInfo, Path>() {
        @Override
        public Path apply(ResourceInfo info) {
          return info.getResource();
        }
      };

  /**
   * @param densities An array of string densities to use for filtering resources
   * @param out The path to use for name spacing the final resource directory.
   * @param working The path of the working directory for the filtering
   */
  public DensitySpecificResourceFilter(List<String> densities, Path out, Path working) {
    this.densities = densities;
    this.out = out;
    this.working = working;
  }

  @VisibleForTesting
  List<Path> getResourceToRemove(List<Path> resourcePaths) {
    List<ResourceInfo> resourceInfos = getResourceInfos(resourcePaths);
    List<ResourceInfo> densityResourceInfos = filterDensityResourceInfos(resourceInfos);
    List<ResourceInfo> resourceInfoToRemove = new ArrayList<>();

    Multimap<String, ResourceInfo> fileGroups = groupResourceInfos(densityResourceInfos,
        GET_RESOURCE_ID);

    for (String key : fileGroups.keySet()) {
      Multimap<String, ResourceInfo> qualifierGroups = groupResourceInfos(fileGroups.get(key),
          GET_RESOURCE_QUALIFIERS);

      for (String qualifiers : qualifierGroups.keySet()) {
        Collection<ResourceInfo> qualifierResourceInfos = qualifierGroups.get(qualifiers);

        if (qualifierResourceInfos.size() != 1) {
          List<ResourceInfo> sortedResourceInfos = Ordering.natural().onResultOf(
              new Function<ResourceInfo, Double>() {
                @Override
                public Double apply(ResourceInfo info) {
                  return matchScore(info, densities);
                }
              }).immutableSortedCopy(qualifierResourceInfos);

          resourceInfoToRemove.addAll(sortedResourceInfos.subList(1, sortedResourceInfos.size()));
        }
      }
    }

    return ImmutableList.copyOf(Lists.transform(resourceInfoToRemove, GET_RESOURCE_PATH));
  }

  private static void removeResources(List<Path> resourceInfoToRemove) {
    for (Path resource : resourceInfoToRemove) {
      resource.toFile().delete();
    }
  }

  private static Multimap<String, ResourceInfo> groupResourceInfos(
      final Collection<ResourceInfo> resourceInfos, Function<ResourceInfo, String> keyFunction) {
    Multimap<String, ResourceInfo> resourceGroups = ArrayListMultimap.create();

    for (ResourceInfo resourceInfo : resourceInfos) {
      resourceGroups.put(keyFunction.apply(resourceInfo), resourceInfo);
    }

    return ImmutableMultimap.copyOf(resourceGroups);
  }

  private static List<ResourceInfo> getResourceInfos(final List<Path> resourcePaths) {
    List<ResourceInfo> resourceInfos = new ArrayList<>();

    for (Path resourcePath : resourcePaths) {
      String qualifiers = resourcePath.getParent().getFileName().toString();
      String density = "";

      for (String densityName : DENSITY_MAP.keySet()) {
        if (qualifiers.contains("-" + densityName)) {
          qualifiers = qualifiers.replace("-" + densityName, "");
          density = densityName;
        }
      }

      String[] qualifierArray = qualifiers.split("-");
      String restype = qualifierArray[0];
      qualifiers = (qualifierArray.length) > 0 ? Joiner.on("-").join(Arrays.copyOfRange(
          qualifierArray, 1, qualifierArray.length)) : "";
      resourceInfos.add(new ResourceInfo(resourcePath, restype, qualifiers, density,
          resourcePath.getFileName().toString()));
    }

    return ImmutableList.copyOf(resourceInfos);
  }

  private static List<ResourceInfo> filterDensityResourceInfos(
      final List<ResourceInfo> resourceInfos) {
    List<ResourceInfo> densityResourceInfos = new ArrayList<>();

    for (ResourceInfo info : resourceInfos) {
      if (info.getRestype().equals("drawable") && !info.getDensity().equals("")
          && !info.getDensity().equals("nodpi") && !info.getResid().endsWith(".xml")) {
        densityResourceInfos.add(info);
      }
    }

    return ImmutableList.copyOf(densityResourceInfos);
  }

  private static double matchScore(ResourceInfo resource, List<String> densities) {
    double score = 0;
    for (String density : densities) {
      score += computeAffinity(DENSITY_MAP.get(resource.getDensity()), DENSITY_MAP.get(density));
    }
    return score;
  }

  private static double computeAffinity(int resourceDensity, int density) {
    if (resourceDensity == density) {
      // Exact match is the best.
      return -2;
    } else if (resourceDensity == 2 * density) {
      // It's very efficient to downsample an image that's exactly 2x the screen
      // density, so we prefer that over other non-perfect matches.
      return -1;
    } else {
      double affinity = Math.log((double) density / resourceDensity) / Math.log(2);

      // We give a slight bump to images that have the same multiplier but are higher quality.
      if (affinity < 0) {
        affinity = Math.abs(affinity) - 0.01;
      }
      return affinity;
    }
  }

  /** Filters the contents of a resource directory. */
  public Path filter(Path unFilteredResourceDir) {
    // no densities to filter, so skip.
    if (densities.isEmpty()) { 
      return unFilteredResourceDir;
    }
    final Path filteredResourceDir =
        out.resolve(working.relativize(unFilteredResourceDir));
    RecursiveFileCopier fileVisitor =
        new RecursiveFileCopier(filteredResourceDir, unFilteredResourceDir);
    try {
      Files.walkFileTree(unFilteredResourceDir, EnumSet.of(FileVisitOption.FOLLOW_LINKS),
          Integer.MAX_VALUE, fileVisitor);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    removeResources(getResourceToRemove(fileVisitor.getCopiedFiles()));
    return filteredResourceDir;
  }
}
