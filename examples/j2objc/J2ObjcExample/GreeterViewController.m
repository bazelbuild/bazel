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

#import "GreeterViewController.h"

#import "examples/j2objc/src/main/java/com/example/myproject/SimpleGreeter.h"

@interface GreeterViewController ()

@property(strong, nonatomic) MyProjectSimpleGreeter *myGreeter;

@property(weak, nonatomic) UILabel *greeterLabel;

@end

@implementation GreeterViewController

- (void)viewDidLoad {
  [super viewDidLoad];

  UILabel *label = [[UILabel alloc] initWithFrame:CGRectMake(10, 10, 200, 40)];
  [self.view addSubview:label];
  _greeterLabel = label;

  _myGreeter = [[MyProjectSimpleGreeter alloc] initWithId:@"world"];
  [self greet];
}

- (void)greet {
  self.greeterLabel.textColor = [UIColor whiteColor];
  self.greeterLabel.text = [_myGreeter hello];
}

@end
