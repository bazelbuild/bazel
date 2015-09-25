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

#import "Equation.h"

@implementation Equation {
  NSMutableArray *_children;
}

@synthesize operation = _operation;

- (id)init {
  [self doesNotRecognizeSelector:_cmd];
  return nil;
}

- (id)initWithOperation:(Operation)operation {
  self = [super init];
  if (self) {
    _children = [NSMutableArray array];
    _operation = operation;
  }
  return self;
}

- (NSArray *)children {
  return [_children copy];
}

- (void)addExpressionAsChild:(Expression *)child {
  [_children addObject:child];
}

- (double)calculate {
  if ([_children count] == 1) {
    return (_operation == kSubtract)
        ? -[_children[0] calculate] : [_children[0] calculate];
  } else if ([_children count] == 0) {
    return (_operation == kSubtract || _operation == kAdd) ? 0 : 1;
  }
  double value = [_children[0] calculate];
  for (Equation *child in [_children subarrayWithRange:
      NSMakeRange(1, [_children count] - 1)]) {
    double childValue = [child calculate];
    switch (_operation) {
      case kAdd:
        value += childValue;
        break;
      case kSubtract:
        value -= childValue;
        break;
      case kMultiply:
        value *= childValue;
        break;
      case kDivide:
        value /= childValue;
        break;
    }
  }
  return value;
}

- (NSString *)description {
  NSMutableString *result = [[NSMutableString alloc] init];
  [result appendString:@"("];
  switch (_operation) {
    case kAdd:
      [result appendString:@"+"];
      break;
    case kSubtract:
      [result appendString:@"-"];
      break;
    case kMultiply:
      [result appendString:@"*"];
      break;
    case kDivide:
      [result appendString:@"/"];
      break;
  }
  for (Equation *child in _children) {
    [result appendString:@" "];
    [result appendString:[child description]];
  }
  [result appendString:@")"];
  return [result copy];
}

- (BOOL)isEqual:(id)object {
  if (![object isKindOfClass:[self class]]) {
    return NO;
  }
  Equation *other = object;
  return other->_operation == _operation &&
      [other->_children isEqual:_children];
}

@end
