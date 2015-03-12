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

#import "Literal.h"

@implementation Literal {
  double _value;
}

- (id)initWithDouble:(double)value {
  self = [super init];
  if (self) {
    _value = value;
  }
  return self;
}

- (double)calculate {
  return _value;
}

- (NSString *)description {
  return [NSString stringWithFormat:@"%0.2lf", _value];
}

- (BOOL)isEqual:(id)object {
  if (![object isKindOfClass:[self class]]) {
    return NO;
  }
  return _value == ((Literal *)object)->_value;
}

@end
