// Copyright 2015 Google Inc. All rights reserved.
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

// Application that finds all Xcodes installed on a given Mac and will return a
// path for a given version number.
//
// If you have 7.0, 6.4.1 and 6.3 installed the inputs will map to:
//
// 7,7.0,7.0.0 = 7.0
// 6,6.4,6.4.1 = 6.4.1
// 6.3,6.3.0 = 6.3

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

#import <CoreServices/CoreServices.h>
#import <Foundation/Foundation.h>

// Simple data structure for tracking a version of Xcode (i.e. 6.4) with an URL
// to the appplication.
@interface XcodeVersionEntry : NSObject
@property(readonly) NSString *version;
@property(readonly) NSURL *url;
@end

@implementation XcodeVersionEntry

- (id)initWithVersion:(NSString *)version url:(NSURL *)url {
  if ((self = [super init])) {
    _version = version;
    _url = url;
  }
  return self;
}

- (id)description {
  return [NSString stringWithFormat:@"<%@ %p>: %@ %@",
                   [self class], self, _version, _url];
}

@end

// Given an entry, insert it into a dictionary that is keyed by versions.
//
// For an entry that is 6.4.1:/Applications/Xcode.app, add it for 6.4.1 and
// optionally add it for 6.4 and 6 if it is "better" than any entry that may
// already be there, where "better" is defined as:
//
// 1. Under /Applications/. (This avoids mounted xcode versions taking
//    precedence over installed versions.)
//
// 2. Not older (at least as high version number).
static void AddEntryToDictionary(
  XcodeVersionEntry *entry,
  NSMutableDictionary<NSString *, XcodeVersionEntry *> *dict) {
  BOOL inApplications = [entry.url.path hasPrefix:@"/Applications/"];
  NSString *entryVersion = entry.version;
  NSString *subversion = entryVersion;
  if (dict[entryVersion] && !inApplications) {
    return;
  }
  dict[entryVersion] = entry;
  while (YES) {
    NSRange range = [subversion rangeOfString:@"." options:NSBackwardsSearch];
    if (range.length == 0 || range.location == 0) {
      break;
    }
    subversion = [subversion substringToIndex:range.location];
    XcodeVersionEntry *subversionEntry = dict[subversion];
    if (subversionEntry) {
      BOOL atLeastAsLarge = ([subversionEntry.version compare:entry.version]
                             == NSOrderedDescending);
      if (inApplications && atLeastAsLarge) {
        dict[subversion] = entry;
      }
    } else {
      dict[subversion] = entry;
    }
  }
}

// Given a "version", expand it to at least 3 components by adding .0 as
// necessary.
static NSString *ExpandVersion(NSString *version) {
  NSArray *components = [version componentsSeparatedByString:@"."];
  NSString *appendage = nil;
  if (components.count == 2) {
    appendage = @".0";
  } else if (components.count == 1) {
    appendage = @".0.0";
  }
  if (appendage) {
    version = [version stringByAppendingString:appendage];
  }
  return version;
}

// Searches for all available Xcodes in the system and returns a dictionary that
// maps version identifiers of any form (X, X.Y, and X.Y.Z) to the directory
// where the Xcode bundle lives.
//
// If there is a problem locating the Xcodes, prints one or more error messages
// and returns nil.
static NSMutableDictionary<NSString *, XcodeVersionEntry *> *FindXcodes()
  __attribute((ns_returns_retained)) {
  CFStringRef cfBundleID = CFSTR("com.apple.dt.Xcode");
  NSString *bundleID = (__bridge NSString *)cfBundleID;

  NSMutableDictionary<NSString *, XcodeVersionEntry *> *dict =
      [[NSMutableDictionary alloc] init];
  CFErrorRef cfError;
  NSArray *array = CFBridgingRelease(LSCopyApplicationURLsForBundleIdentifier(
      cfBundleID, &cfError));
  if (array == nil) {
    NSError *nsError = (__bridge NSError *)cfError;
    fprintf(stderr, "error: %s\n", nsError.description.UTF8String);
    return nil;
  }

  // Scan all bundles but delay returning in case of errors until we are
  // done. This is to let us log details about all the bundles that were
  // processed so that a faulty bundle doesn't hide useful information about
  // other bundles that were found.
  BOOL errors = NO;
  for (NSURL *url in array) {
    NSArray *contents = [
      [NSFileManager defaultManager] contentsOfDirectoryAtURL:url
                                   includingPropertiesForKeys:nil
                                                      options:0
                                                        error:nil];
    NSLog(@"Found bundle %@ in %@; contents on disk: %@",
          bundleID, url, contents);

    NSBundle *bundle = [NSBundle bundleWithURL:url];
    if (bundle == nil) {
      NSLog(@"ERROR: Unable to open bundle at URL: %@\n", url);
      errors = YES;
      continue;
    }

    // LSCopyApplicationURLsForBundleIdentifier seems to sometimes return
    // invalid bundles (e.g. an arbitrary folder), which we should ignore (but
    // don't treat as an error).
    //
    // To work around this issue, we double check to make sure the NSBundle's
    // bundleIdentifier is that of Xcode's, as invalid bundles won't match.
    if (![bundle.bundleIdentifier isEqualToString:bundleID]) {
      NSLog(@"WARNING: Ignoring bundle %@ due to bundleID mismatch "
            @"(got \"%@\" but expected \"%@\"); info: %@",
            url, bundle.bundleIdentifier, bundleID, bundle.infoDictionary);
      continue;
    }

    NSString *versionKey = @"CFBundleShortVersionString";
    NSString *version = [bundle.infoDictionary objectForKey:versionKey];
    if (version == nil) {
      NSLog(@"ERROR: Cannot find %@ in info for bundle %@; info: %@\n",
            versionKey, url, bundle.infoDictionary);
      errors = YES;
      continue;
    }
    NSString *expandedVersion = ExpandVersion(version);
    NSLog(@"Version strings for %@: short=%@, expanded=%@",
          url, version, expandedVersion);

    NSURL *versionPlistUrl = [url URLByAppendingPathComponent:@"Contents/version.plist"];
    NSDictionary *versionPlistContents =
        [[NSDictionary alloc] initWithContentsOfURL:versionPlistUrl];
    NSString *productVersion = [versionPlistContents objectForKey:@"ProductBuildVersion"];
    if (productVersion) {
      expandedVersion = [expandedVersion stringByAppendingFormat:@".%@", productVersion];
    }

    NSURL *developerDir =
        [url URLByAppendingPathComponent:@"Contents/Developer"];
    XcodeVersionEntry *entry =
        [[XcodeVersionEntry alloc] initWithVersion:expandedVersion
                                               url:developerDir];
    AddEntryToDictionary(entry, dict);
  }
  return errors ? nil : dict;
}

// Prints out the located Xcodes as a set of lines where each line contains the
// list of versions for a given Xcode and its location on disk.
static void DumpAsVersionsOnly(
  FILE *output,
  NSMutableDictionary<NSString *, XcodeVersionEntry *> *dict) {
  NSMutableDictionary<NSString *, NSMutableSet <NSString *> *> *aliasDict =
      [[NSMutableDictionary alloc] init];
  [dict enumerateKeysAndObjectsUsingBlock:^(NSString *aliasVersion,
                                            XcodeVersionEntry *entry,
                                            BOOL *stop) {
    NSString *versionString = entry.version;
    if (aliasDict[versionString] == nil) {
      aliasDict[versionString] = [[NSMutableSet alloc] init];
    }
    [aliasDict[versionString] addObject:aliasVersion];
  }];
  for (NSString *version in aliasDict) {
    XcodeVersionEntry *entry = dict[version];
    fprintf(output, "%s:%s:%s\n",
            version.UTF8String,
            [[aliasDict[version] allObjects]
                   componentsJoinedByString: @","].UTF8String,
            entry.url.fileSystemRepresentation);
  }
}

// Prints out the located Xcodes in JSON format.
static void DumpAsJson(
  FILE *output,
  NSMutableDictionary<NSString *, XcodeVersionEntry *> *dict) {
  fprintf(output, "{\n");
  for (NSString *version in dict) {
    XcodeVersionEntry *entry = dict[version];
    fprintf(output, "\t\"%s\": \"%s\",\n",
            version.UTF8String, entry.url.fileSystemRepresentation);
  }
  fprintf(output, "}\n");
}

// Dumps usage information.
static void usage(FILE *output) {
  fprintf(
      output,
      "xcode-locator [-v|<version_number>]"
      "\n\n"
      "Given a version number or partial version number in x.y.z format, "
      "will attempt to return the path to the appropriate developer "
      "directory."
      "\n\n"
      "Omitting a version number will list all available versions in JSON "
      "format, alongside their paths."
      "\n\n"
      "Passing -v will list all available fully-specified version numbers "
      "along with their possible aliases and their developer directory, "
      "each on a new line. For example:"
      "\n\n"
      "7.3.1:7,7.3,7.3.1:/Applications/Xcode.app/Contents/Developer"
      "\n");
}

int main(int argc, const char * argv[]) {
  @autoreleasepool {
    NSString *versionArg = nil;
    BOOL versionsOnly = NO;
    if (argc == 1) {
      versionArg = @"";
    } else if (argc == 2) {
      NSString *firstArg = [NSString stringWithUTF8String:argv[1]];
      if ([@"-v" isEqualToString:firstArg]) {
        versionsOnly = YES;
        versionArg = @"";
      } else {
        versionArg = firstArg;
      }
    }
    if (versionArg == nil) {
      usage(stderr);
      return 1;
    }

    NSMutableDictionary<NSString *, XcodeVersionEntry *> *dict = FindXcodes();
    if (dict == nil) {
      return 1;
    }

    XcodeVersionEntry *entry = [dict objectForKey:versionArg];
    if (entry) {
      printf("%s\n", entry.url.fileSystemRepresentation);
      return 0;
    }

    if (versionsOnly) {
      DumpAsVersionsOnly(stdout, dict);
    } else {
      DumpAsJson(stdout, dict);
    }
    return ([@"" isEqualToString:versionArg] ? 0 : 1);
  }
}
