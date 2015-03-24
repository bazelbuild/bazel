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

#import "CalculatedValues.h"

#import "Equation.h"

@implementation CalculatedValues {
  NSMutableArray *_values;
}

+ (CalculatedValues *)sharedInstance {
  static CalculatedValues *values = nil;
  if (!values) {
    values = [[[self class] alloc] init];
  }
  return values;
}

- (id)init {
  self = [super init];
  if (self) {
    _values = [NSMutableArray array];
  }
  return self;
}

- (NSArray *)values {
  return [_values copy];
}

- (void)addEquation:(Equation *)equation {
  [_values addObject:
      [NSString stringWithFormat:@"%@ = %0.2lf", equation, [equation calculate]]];
}

@end
