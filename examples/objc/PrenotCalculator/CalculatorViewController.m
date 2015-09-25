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

#import "CalculatorViewController.h"

#import "CalculatedValues.h"
#import "CoreData.h"
#import "Equation.h"
#import "Expression.h"
#import "Literal.h"

@interface CalculatorViewController ()
- (void)updateEnabledViews;
- (void)updateButtonColor:(UIButton *)button;
@end

@implementation CalculatorViewController {
  NSMutableArray *_locationStack;
  NSMutableString *_currentNumber;
}

- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil {
  self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
  if (self) {
    self.title = @"Calculator";
    CoreData *coreData = [[CoreData alloc] init];
    [coreData verify];
  }
  return self;
}

- (void)viewDidLoad {
  [super viewDidLoad];
  _locationStack = [NSMutableArray array];
  _currentNumber = [NSMutableString stringWithString:@""];
  [self updateEnabledViews];
}

- (IBAction)enterDigit:(id)sender {
  UIButton *button = (UIButton *)sender;
  [_currentNumber appendString:button.titleLabel.text];
  self.resultLabel.text = _currentNumber;
  [self updateEnabledViews];
}

- (IBAction)finish:(id)sender {
  if (![_locationStack count]) {
    _currentNumber = [NSMutableString stringWithString:@""];
    self.resultLabel.text = @"hello";
  }
  if (![_locationStack count]) {
    _currentNumber = [NSMutableString stringWithString:@""];
  } else if ([_currentNumber length]) {
    Equation *last = [_locationStack lastObject];
    [last addExpressionAsChild:[[Literal alloc] initWithDouble:
        atof([_currentNumber cStringUsingEncoding:NSUTF8StringEncoding])]];
    _currentNumber = [NSMutableString stringWithString:@""];
    self.resultLabel.text = [NSString stringWithFormat:@"%@", last];
  } else {
    Equation *popped = [_locationStack lastObject];
    [_locationStack removeLastObject];
    [[CalculatedValues sharedInstance] addEquation:popped];
    self.resultLabel.text =
        [NSString stringWithFormat:@"%lf", [popped calculate]];
  }
  [self updateEnabledViews];
}

- (IBAction)operate:(id)sender {
  Operation operation;
  if ([_currentNumber length]) {
    // finish the current number first automatically:
    [self finish:nil];
  }
  if (sender == self.multiplyButton) {
    operation = kMultiply;
  } else if (sender == self.plusButton) {
    operation = kAdd;
  } else if (sender == self.divideButton) {
    operation = kDivide;
  } else if (sender == self.minusButton) {
    operation = kSubtract;
  } else {
    // Shouldn't happen
    operation = kAdd;
  }
  Equation *newEquation = [[Equation alloc] initWithOperation:operation];
  if ([_locationStack count]) {
    [[_locationStack lastObject] addExpressionAsChild:newEquation];
  }
  [_locationStack addObject:newEquation];
  self.resultLabel.text = [NSString stringWithFormat:@"%@", _locationStack[0]];
  [self updateEnabledViews];
}

#pragma mark - Private


- (void)updateEnabledViews {
  self.finishButton.enabled = [_locationStack count] || [_currentNumber length];
  [self updateButtonColor:self.finishButton];
  self.zeroButton.enabled = self.finishButton.enabled;
  [self updateButtonColor:self.zeroButton];
  BOOL enableNonZeroDigits = ![_currentNumber isEqualToString:@"0"];
  for (UIButton *digit in self.nonZeroDigitButtons) {
    digit.enabled = enableNonZeroDigits;
    [self updateButtonColor:digit];
  }
}

- (void)updateButtonColor:(UIButton *)button {
  button.backgroundColor = button.enabled ?
      [UIColor whiteColor] : [UIColor grayColor];
}

@end
