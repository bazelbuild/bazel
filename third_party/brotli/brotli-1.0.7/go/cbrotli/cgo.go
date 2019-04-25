// Copyright 2017 Google Inc. All Rights Reserved.
//
// Distributed under MIT license.
// See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

package cbrotli

// Inform golang build system that it should link brotli libraries.

// #cgo LDFLAGS: -lbrotlicommon
// #cgo LDFLAGS: -lbrotlidec
// #cgo LDFLAGS: -lbrotlienc
import "C"
