// Copyright 2016 Google Inc. All Rights Reserved.
//
// Distributed under MIT license.
// See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

package cbrotli

/*
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <brotli/encode.h>

struct CompressStreamResult {
  size_t bytes_consumed;
  const uint8_t* output_data;
  size_t output_data_size;
  int success;
  int has_more;
};

static struct CompressStreamResult CompressStream(
    BrotliEncoderState* s, BrotliEncoderOperation op,
    const uint8_t* data, size_t data_size) {
  struct CompressStreamResult result;
  size_t available_in = data_size;
  const uint8_t* next_in = data;
  size_t available_out = 0;
  result.success = BrotliEncoderCompressStream(s, op,
      &available_in, &next_in, &available_out, 0, 0) ? 1 : 0;
  result.bytes_consumed = data_size - available_in;
  result.output_data = 0;
  result.output_data_size = 0;
  if (result.success) {
    result.output_data = BrotliEncoderTakeOutput(s, &result.output_data_size);
  }
  result.has_more = BrotliEncoderHasMoreOutput(s) ? 1 : 0;
  return result;
}
*/
import "C"

import (
	"bytes"
	"errors"
	"io"
	"unsafe"
)

// WriterOptions configures Writer.
type WriterOptions struct {
	// Quality controls the compression-speed vs compression-density trade-offs.
	// The higher the quality, the slower the compression. Range is 0 to 11.
	Quality int
	// LGWin is the base 2 logarithm of the sliding window size.
	// Range is 10 to 24. 0 indicates automatic configuration based on Quality.
	LGWin int
}

// Writer implements io.WriteCloser by writing Brotli-encoded data to an
// underlying Writer.
type Writer struct {
	dst          io.Writer
	state        *C.BrotliEncoderState
	buf, encoded []byte
}

var (
	errEncode       = errors.New("cbrotli: encode error")
	errWriterClosed = errors.New("cbrotli: Writer is closed")
)

// NewWriter initializes new Writer instance.
// Close MUST be called to free resources.
func NewWriter(dst io.Writer, options WriterOptions) *Writer {
	state := C.BrotliEncoderCreateInstance(nil, nil, nil)
	C.BrotliEncoderSetParameter(
		state, C.BROTLI_PARAM_QUALITY, (C.uint32_t)(options.Quality))
	if options.LGWin > 0 {
		C.BrotliEncoderSetParameter(
			state, C.BROTLI_PARAM_LGWIN, (C.uint32_t)(options.LGWin))
	}
	return &Writer{
		dst:   dst,
		state: state,
	}
}

func (w *Writer) writeChunk(p []byte, op C.BrotliEncoderOperation) (n int, err error) {
	if w.state == nil {
		return 0, errWriterClosed
	}

	for {
		var data *C.uint8_t
		if len(p) != 0 {
			data = (*C.uint8_t)(&p[0])
		}
		result := C.CompressStream(w.state, op, data, C.size_t(len(p)))
		if result.success == 0 {
			return n, errEncode
		}
		p = p[int(result.bytes_consumed):]
		n += int(result.bytes_consumed)

		length := int(result.output_data_size)
		if length != 0 {
			// It is a workaround for non-copying-wrapping of native memory.
			// C-encoder never pushes output block longer than ((2 << 25) + 502).
			// TODO: use natural wrapper, when it becomes available, see
			//               https://golang.org/issue/13656.
			output := (*[1 << 30]byte)(unsafe.Pointer(result.output_data))[:length:length]
			_, err = w.dst.Write(output)
			if err != nil {
				return n, err
			}
		}
		if len(p) == 0 && result.has_more == 0 {
			return n, nil
		}
	}
}

// Flush outputs encoded data for all input provided to Write. The resulting
// output can be decoded to match all input before Flush, but the stream is
// not yet complete until after Close.
// Flush has a negative impact on compression.
func (w *Writer) Flush() error {
	_, err := w.writeChunk(nil, C.BROTLI_OPERATION_FLUSH)
	return err
}

// Close flushes remaining data to the decorated writer and frees C resources.
func (w *Writer) Close() error {
	// If stream is already closed, it is reported by `writeChunk`.
	_, err := w.writeChunk(nil, C.BROTLI_OPERATION_FINISH)
	// C-Brotli tolerates `nil` pointer here.
	C.BrotliEncoderDestroyInstance(w.state)
	w.state = nil
	return err
}

// Write implements io.Writer. Flush or Close must be called to ensure that the
// encoded bytes are actually flushed to the underlying Writer.
func (w *Writer) Write(p []byte) (n int, err error) {
	return w.writeChunk(p, C.BROTLI_OPERATION_PROCESS)
}

// Encode returns content encoded with Brotli.
func Encode(content []byte, options WriterOptions) ([]byte, error) {
	var buf bytes.Buffer
	writer := NewWriter(&buf, options)
	_, err := writer.Write(content)
	if closeErr := writer.Close(); err == nil {
		err = closeErr
	}
	return buf.Bytes(), err
}
