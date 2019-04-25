// Copyright 2016 Google Inc. All Rights Reserved.
//
// Distributed under MIT license.
// See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

// Package cbrotli compresses and decompresses data with C-Brotli library.
package cbrotli

/*
#include <stddef.h>
#include <stdint.h>

#include <brotli/decode.h>

static BrotliDecoderResult DecompressStream(BrotliDecoderState* s,
                                            uint8_t* out, size_t out_len,
                                            const uint8_t* in, size_t in_len,
                                            size_t* bytes_written,
                                            size_t* bytes_consumed) {
  size_t in_remaining = in_len;
  size_t out_remaining = out_len;
  BrotliDecoderResult result = BrotliDecoderDecompressStream(
      s, &in_remaining, &in, &out_remaining, &out, NULL);
  *bytes_written = out_len - out_remaining;
  *bytes_consumed = in_len - in_remaining;
  return result;
}
*/
import "C"

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
)

type decodeError C.BrotliDecoderErrorCode

func (err decodeError) Error() string {
	return "cbrotli: " +
		C.GoString(C.BrotliDecoderErrorString(C.BrotliDecoderErrorCode(err)))
}

var errExcessiveInput = errors.New("cbrotli: excessive input")
var errInvalidState = errors.New("cbrotli: invalid state")
var errReaderClosed = errors.New("cbrotli: Reader is closed")

// Reader implements io.ReadCloser by reading Brotli-encoded data from an
// underlying Reader.
type Reader struct {
	src   io.Reader
	state *C.BrotliDecoderState
	buf   []byte // scratch space for reading from src
	in    []byte // current chunk to decode; usually aliases buf
}

// readBufSize is a "good" buffer size that avoids excessive round-trips
// between C and Go but doesn't waste too much memory on buffering.
// It is arbitrarily chosen to be equal to the constant used in io.Copy.
const readBufSize = 32 * 1024

// NewReader initializes new Reader instance.
// Close MUST be called to free resources.
func NewReader(src io.Reader) *Reader {
	return &Reader{
		src:   src,
		state: C.BrotliDecoderCreateInstance(nil, nil, nil),
		buf:   make([]byte, readBufSize),
	}
}

// Close implements io.Closer. Close MUST be invoked to free native resources.
func (r *Reader) Close() error {
	if r.state == nil {
		return errReaderClosed
	}
	// Close despite the state; i.e. there might be some unread decoded data.
	C.BrotliDecoderDestroyInstance(r.state)
	r.state = nil
	return nil
}

func (r *Reader) Read(p []byte) (n int, err error) {
	if int(C.BrotliDecoderHasMoreOutput(r.state)) == 0 && len(r.in) == 0 {
		m, readErr := r.src.Read(r.buf)
		if m == 0 {
			// If readErr is `nil`, we just proxy underlying stream behavior.
			return 0, readErr
		}
		r.in = r.buf[:m]
	}

	if len(p) == 0 {
		return 0, nil
	}

	for {
		var written, consumed C.size_t
		var data *C.uint8_t
		if len(r.in) != 0 {
			data = (*C.uint8_t)(&r.in[0])
		}
		result := C.DecompressStream(r.state,
			(*C.uint8_t)(&p[0]), C.size_t(len(p)),
			data, C.size_t(len(r.in)),
			&written, &consumed)
		r.in = r.in[int(consumed):]
		n = int(written)

		switch result {
		case C.BROTLI_DECODER_RESULT_SUCCESS:
			if len(r.in) > 0 {
				return n, errExcessiveInput
			}
			return n, nil
		case C.BROTLI_DECODER_RESULT_ERROR:
			return n, decodeError(C.BrotliDecoderGetErrorCode(r.state))
		case C.BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT:
			if n == 0 {
				return 0, io.ErrShortBuffer
			}
			return n, nil
		case C.BROTLI_DECODER_NEEDS_MORE_INPUT:
		}

		if len(r.in) != 0 {
			return 0, errInvalidState
		}

		// Calling r.src.Read may block. Don't block if we have data to return.
		if n > 0 {
			return n, nil
		}

		// Top off the buffer.
		encN, err := r.src.Read(r.buf)
		if encN == 0 {
			// Not enough data to complete decoding.
			if err == io.EOF {
				return 0, io.ErrUnexpectedEOF
			}
			return 0, err
		}
		r.in = r.buf[:encN]
	}

	return n, nil
}

// Decode decodes Brotli encoded data.
func Decode(encodedData []byte) ([]byte, error) {
	r := &Reader{
		src:   bytes.NewReader(nil),
		state: C.BrotliDecoderCreateInstance(nil, nil, nil),
		buf:   make([]byte, 4), // arbitrarily small but nonzero so that r.src.Read returns io.EOF
		in:    encodedData,
	}
	defer r.Close()
	return ioutil.ReadAll(r)
}
